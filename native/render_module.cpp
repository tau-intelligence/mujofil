// mujofil-warp native module, milestone 5a:
//   A Python-callable Filament renderer that runs on OUR shared Vulkan device
//   (external-memory enabled) and hands PyTorch a CUDA tensor of the rendered
//   pixels with NO CPU round-trip (DLPack).
//
// This is the productionized union of two proven spikes:
//   - spikes/filament_vk_export/render_export.cpp (shared device + exportable
//     headless swapchain + render + Vulkan->CUDA export)
//   - spikes/dlpack_torch/dlpack_cuda.cpp (CUDA ptr -> torch via DLPack)
//
// 5a renders a clear color (no scene yet) to prove the end-to-end Python path.
// 5b wires in mujofil's SceneBridge for real MuJoCo geometry.
//
// mujofil's PyPI release is untouched: this is a SEPARATE .so built in this repo.

// Do NOT include <vulkan/vulkan.h> directly — BlueVK provides the Vk* symbols.
#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/Viewport.h>
#include <filament/SwapChain.h>
#include <backend/platforms/VulkanPlatform.h>
#include <utils/EntityManager.h>

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <pybind11/pybind11.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace filament;
using namespace filament::backend;
using namespace bluevk;

namespace {

constexpr uint32_t SC_SIZE = 2;
constexpr VkFormat COLOR_FMT = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat DEPTH_FMT = VK_FORMAT_D32_SFLOAT;

void vkcheck(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what);
}

uint32_t pickMemType(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool hasExt(const std::vector<VkExtensionProperties>& a, const char* n) {
    for (auto& e : a) if (!strcmp(e.extensionName, n)) return true;
    return false;
}

// ----- exportable headless swapchain (from the proven spike) ----------------
struct ExportSwapChain : public Platform::SwapChain {
    VulkanPlatform::SwapChainBundle bundle;
    std::vector<VkDeviceMemory> colorMem;
    VkDeviceMemory depthMem = VK_NULL_HANDLE;
};

class ExportPlatform : public VulkanPlatform {
public:
    ExportSwapChain* sc = nullptr;
    uint32_t lastAcquired = 0;

    SwapChainPtr createSwapChain(void*, uint64_t, VkExtent2D extent) override {
        VkDevice device = getDevice();
        VkPhysicalDevice phys = getPhysicalDevice();
        auto* s = new ExportSwapChain();
        s->bundle.extent = extent;
        s->bundle.colorFormat = COLOR_FMT;
        s->bundle.depthFormat = DEPTH_FMT;
        s->bundle.colors.reserve(SC_SIZE);
        s->bundle.colors.resize(SC_SIZE);
        s->colorMem.resize(SC_SIZE);
        for (uint32_t i = 0; i < SC_SIZE; ++i) {
            VkExternalMemoryImageCreateInfo ext{
                VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
            ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ci.pNext = &ext;
            ci.imageType = VK_IMAGE_TYPE_2D;
            ci.format = COLOR_FMT;
            ci.extent = {extent.width, extent.height, 1};
            ci.mipLevels = 1; ci.arrayLayers = 1;
            ci.samples = VK_SAMPLE_COUNT_1_BIT;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL;
            ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            vkcheck(vkCreateImage(device, &ci, nullptr, &s->bundle.colors[i]), "createImage");
            VkMemoryRequirements req;
            vkGetImageMemoryRequirements(device, s->bundle.colors[i], &req);
            uint32_t mt = pickMemType(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            ded.image = s->bundle.colors[i];
            VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
            exp.pNext = &ded; exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &exp; ai.allocationSize = req.size; ai.memoryTypeIndex = mt;
            vkcheck(vkAllocateMemory(device, &ai, nullptr, &s->colorMem[i]), "allocMem");
            vkcheck(vkBindImageMemory(device, s->bundle.colors[i], s->colorMem[i], 0), "bind");
        }
        VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dci.imageType = VK_IMAGE_TYPE_2D; dci.format = DEPTH_FMT;
        dci.extent = {extent.width, extent.height, 1};
        dci.mipLevels = 1; dci.arrayLayers = 1; dci.samples = VK_SAMPLE_COUNT_1_BIT;
        dci.tiling = VK_IMAGE_TILING_OPTIMAL;
        dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        dci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        dci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkcheck(vkCreateImage(device, &dci, nullptr, &s->bundle.depth), "depthImage");
        VkMemoryRequirements dreq; vkGetImageMemoryRequirements(device, s->bundle.depth, &dreq);
        uint32_t dmt = pickMemType(phys, dreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dreq.size; dai.memoryTypeIndex = dmt;
        vkcheck(vkAllocateMemory(device, &dai, nullptr, &s->depthMem), "depthMem");
        vkcheck(vkBindImageMemory(device, s->bundle.depth, s->depthMem, 0), "depthBind");
        sc = s;
        return s;
    }
    SwapChainBundle getSwapChainBundle(SwapChainPtr h) override {
        return ((ExportSwapChain*)h)->bundle;
    }
    VkResult acquire(SwapChainPtr, ImageSyncData* out) override {
        out->imageIndex = lastAcquired;
        out->imageReadySemaphore = VK_NULL_HANDLE;
        out->explicitImageReadyWait = nullptr;
        lastAcquired = (lastAcquired + 1) % SC_SIZE;
        return VK_SUCCESS;
    }
    VkResult present(SwapChainPtr, uint32_t, VkSemaphore) override { return VK_SUCCESS; }
    bool hasResized(SwapChainPtr) override { return false; }
    bool isProtected(SwapChainPtr) override { return false; }
    VkResult recreate(SwapChainPtr) override { return VK_SUCCESS; }
    void destroy(SwapChainPtr h) override {
        auto* s = (ExportSwapChain*)h;
        VkDevice device = getDevice();
        for (uint32_t i = 0; i < s->bundle.colors.size(); ++i) {
            vkDestroyImage(device, s->bundle.colors[i], nullptr);
            vkFreeMemory(device, s->colorMem[i], nullptr);
        }
        vkDestroyImage(device, s->bundle.depth, nullptr);
        vkFreeMemory(device, s->depthMem, nullptr);
        delete s;
    }
};

// ----- DLPack export of a CUDA buffer ---------------------------------------
struct CudaBlob {
    void* dptr = nullptr;
    cudaExternalMemory_t ext = nullptr;
    int64_t shape[3];
    int device = 0;
};

void dlpack_deleter(DLManagedTensor* self) {
    auto* b = static_cast<CudaBlob*>(self->manager_ctx);
    if (b) { /* memory owned by the renderer; do not free here */ delete b; }
    delete self;
}
void capsule_destructor(PyObject* cap) {
    if (PyCapsule_IsValid(cap, "dltensor")) {
        auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (mt && mt->deleter) mt->deleter(mt);
    }
}

// ----- the renderer ---------------------------------------------------------
class WarpRenderer {
public:
    WarpRenderer(uint32_t width, uint32_t height) : W_(width), H_(height) {
        initVulkan();
        initFilament();
    }
    ~WarpRenderer() {
        if (cudaDptr_) cudaFree(cudaDptr_);
        if (cudaExt_) cudaDestroyExternalMemory(cudaExt_);
        if (engine_) Engine::destroy(&engine_);
    }

    void set_clear_color(float r, float g, float b) { clear_ = {r, g, b, 1.0f}; }

    // Render a frame and return a DLPack capsule (H,W,4) uint8 cuda tensor.
    py::capsule render_dlpack() {
        Renderer::ClearOptions co; co.clearColor = clear_; co.clear = true;
        renderer_->setClearOptions(co);
        if (renderer_->beginFrame(swapchain_)) {
            renderer_->render(view_);
            renderer_->endFrame();
        }
        engine_->flushAndWait();
        copyToCuda();   // GPU image -> exportable buffer -> CUDA ptr

        auto* blob = new CudaBlob();
        blob->dptr = cudaDptr_;
        blob->device = cudaDevice_;
        blob->shape[0] = H_; blob->shape[1] = W_; blob->shape[2] = 4;
        auto* mt = new DLManagedTensor();
        mt->manager_ctx = blob;
        mt->deleter = dlpack_deleter;
        DLTensor& t = mt->dl_tensor;
        t.data = cudaDptr_;
        t.device = DLDevice{kDLCUDA, cudaDevice_};
        t.ndim = 3;
        t.dtype = DLDataType{kDLUInt, 8, 1};
        t.shape = blob->shape;
        t.strides = nullptr;
        t.byte_offset = 0;
        return py::capsule(mt, "dltensor", capsule_destructor);
    }

    uint32_t width() const { return W_; }
    uint32_t height() const { return H_; }

private:
    void initVulkan() {
        if (!bluevk::initialize()) throw std::runtime_error("bluevk init");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        vkcheck(vkCreateInstance(&ici, nullptr, &instance_), "createInstance");
        bluevk::bindInstance(instance_);

        uint32_t nd = 0; vkEnumeratePhysicalDevices(instance_, &nd, nullptr);
        std::vector<VkPhysicalDevice> devs(nd);
        vkEnumeratePhysicalDevices(instance_, &nd, devs.data());
        for (auto d : devs) {
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
            if (p.vendorID == 0x10DE) { phys_ = d; break; }
        }
        if (!phys_) throw std::runtime_error("no NVIDIA Vulkan device");

        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qfs.data());
        uint32_t qcount = 0;
        for (uint32_t i = 0; i < qn; ++i)
            if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfi_ = i; qcount = qfs[i].queueCount; break; }
        uint32_t nQ = qcount >= 2 ? 2 : 1;
        filQueueIdx_ = nQ == 2 ? 1 : 0;
        float prio[2] = {1.0f, 1.0f};

        uint32_t en = 0; vkEnumerateDeviceExtensionProperties(phys_, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> avail(en);
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &en, avail.data());
        const char* wants[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE1_EXTENSION_NAME,
            VK_KHR_MAINTENANCE2_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME,
            VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
        std::vector<const char*> enable;
        for (auto* e : wants) if (hasExt(avail, e)) enable.push_back(e);

        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfi_; qci.queueCount = nQ; qci.pQueuePriorities = prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)enable.size();
        dci.ppEnabledExtensionNames = enable.data();
        vkcheck(vkCreateDevice(phys_, &dci, nullptr, &device_), "createDevice");
        vkGetDeviceQueue(device_, qfi_, 0, &myQueue_);

        pfnGetMemoryFd_ = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        if (!pfnGetMemoryFd_) throw std::runtime_error("vkGetMemoryFdKHR unavailable");

        // match the CUDA device to this Vulkan device by UUID
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &idp; vkGetPhysicalDeviceProperties2(phys_, &p2);
        int n = 0; cudaGetDeviceCount(&n);
        for (int i = 0; i < n; ++i) {
            cudaDeviceProp cp; cudaGetDeviceProperties(&cp, i);
            if (memcmp(cp.uuid.bytes, idp.deviceUUID, 16) == 0) { cudaDevice_ = i; break; }
        }
        cudaSetDevice(cudaDevice_);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = qfi_;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkcheck(vkCreateCommandPool(device_, &pci, nullptr, &pool_), "pool");
    }

    void initFilament() {
        VulkanPlatform::VulkanSharedContext ctx{};
        ctx.instance = instance_; ctx.physicalDevice = phys_; ctx.logicalDevice = device_;
        ctx.graphicsQueueFamilyIndex = qfi_; ctx.graphicsQueueIndex = filQueueIdx_;
        engine_ = Engine::Builder()
                .backend(Backend::VULKAN).platform(&platform_).sharedContext(&ctx)
                .featureLevel(Engine::FeatureLevel::FEATURE_LEVEL_1).build();
        if (!engine_) throw std::runtime_error("Filament engine build failed");
        swapchain_ = engine_->createSwapChain(W_, H_);
        renderer_ = engine_->createRenderer();
        scene_ = engine_->createScene();
        view_ = engine_->createView();
        camEnt_ = utils::EntityManager::get().create();
        camera_ = engine_->createCamera(camEnt_);
        view_->setScene(scene_);
        view_->setCamera(camera_);
        view_->setViewport({0, 0, W_, H_});
        view_->setPostProcessingEnabled(false);
    }

    void copyToCuda() {
        const VkDeviceSize bytes = (VkDeviceSize)W_ * H_ * 4;
        uint32_t idx = (platform_.lastAcquired + SC_SIZE - 1) % SC_SIZE;
        VkImage rendered = platform_.sc->bundle.colors[idx];

        if (!expBuf_) {
            VkExternalMemoryBufferCreateInfo extBuf{
                VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            extBuf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.pNext = &extBuf; bci.size = bytes;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vkcheck(vkCreateBuffer(device_, &bci, nullptr, &expBuf_), "expBuf");
            VkMemoryRequirements req; vkGetBufferMemoryRequirements(device_, expBuf_, &req);
            uint32_t mt = pickMemType(phys_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            ded.buffer = expBuf_;
            VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
            exp.pNext = &ded; exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &exp; ai.allocationSize = req.size; ai.memoryTypeIndex = mt;
            vkcheck(vkAllocateMemory(device_, &ai, nullptr, &expMem_), "expMem");
            vkcheck(vkBindBufferMemory(device_, expBuf_, expMem_, 0), "expBind");

            VkMemoryGetFdInfoKHR gfd{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
            gfd.memory = expMem_; gfd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            int fd = -1; vkcheck(pfnGetMemoryFd_(device_, &gfd, &fd), "getFd");
            cudaExternalMemoryHandleDesc hd{};
            hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
            hd.handle.fd = fd; hd.size = req.size; hd.flags = cudaExternalMemoryDedicated;
            if (cudaImportExternalMemory(&cudaExt_, &hd) != cudaSuccess)
                throw std::runtime_error("cudaImportExternalMemory");
            cudaExternalMemoryBufferDesc bd{}; bd.offset = 0; bd.size = bytes; bd.flags = 0;
            if (cudaExternalMemoryGetMappedBuffer(&cudaDptr_, cudaExt_, &bd) != cudaSuccess)
                throw std::runtime_error("cudaExternalMemoryGetMappedBuffer");

            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
            vkcheck(vkAllocateCommandBuffers(device_, &cai, &cmd_), "cmd");
        }

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
        VkImageMemoryBarrier toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toSrc.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image = rendered;
        toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toSrc.srcAccessMask = 0; toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSrc);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {W_, H_, 1};
        vkCmdCopyImageToBuffer(cmd_, rendered, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               expBuf_, 1, &region);
        vkEndCommandBuffer(cmd_);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd_;
        vkQueueSubmit(myQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(myQueue_);   // 5a: coarse sync; later a VK->CUDA semaphore
    }

    uint32_t W_, H_;
    math::float4 clear_ = {0.2f, 0.6f, 0.9f, 1.0f};

    // Vulkan
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue myQueue_ = VK_NULL_HANDLE;
    uint32_t qfi_ = 0, filQueueIdx_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdKHR pfnGetMemoryFd_ = nullptr;
    VkBuffer expBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory expMem_ = VK_NULL_HANDLE;

    // Filament
    ExportPlatform platform_;
    Engine* engine_ = nullptr;
    filament::SwapChain* swapchain_ = nullptr;
    Renderer* renderer_ = nullptr;
    Scene* scene_ = nullptr;
    View* view_ = nullptr;
    Camera* camera_ = nullptr;
    utils::Entity camEnt_;

    // CUDA
    int cudaDevice_ = 0;
    cudaExternalMemory_t cudaExt_ = nullptr;
    void* cudaDptr_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(_mujofil_warp, m) {
    m.doc() = "mujofil-warp native: Filament render -> torch CUDA tensor (zero-copy)";
    py::class_<WarpRenderer>(m, "WarpRenderer")
        .def(py::init<uint32_t, uint32_t>(), py::arg("width"), py::arg("height"))
        .def("set_clear_color", &WarpRenderer::set_clear_color)
        .def("render_dlpack", &WarpRenderer::render_dlpack,
             "Render a frame; return a DLPack capsule (H,W,4 uint8 cuda) for "
             "torch.from_dlpack — no CPU round-trip.")
        .def_property_readonly("width", &WarpRenderer::width)
        .def_property_readonly("height", &WarpRenderer::height);
}
