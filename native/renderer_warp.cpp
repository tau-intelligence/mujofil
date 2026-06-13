// Implementation of vf_mujoco::Renderer for mujofil-warp: shared Vulkan device +
// exportable headless swapchain + CUDA export. All bluevk/Vulkan/CUDA lives here
// (PIMPL) so the vendored SceneBridge TU stays free of bluevk.
//
// Do NOT include <vulkan/vulkan.h> — BlueVK provides the Vk* symbols.
#include "core/renderer.h"

#include <backend/platforms/VulkanPlatform.h>
#include <filament/Renderer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <filament/Options.h>
#include <utils/EntityManager.h>

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <vector>

using namespace filament;
using namespace filament::backend;
using namespace bluevk;

namespace vf_mujoco {

namespace {
constexpr uint32_t SC_SIZE = 2;
constexpr VkFormat COLOR_FMT = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat DEPTH_FMT = VK_FORMAT_D32_SFLOAT;

void vkok(VkResult r, const char* what) {
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
            ci.pNext = &ext; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = COLOR_FMT;
            ci.extent = {extent.width, extent.height, 1};
            ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL;
            ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            vkok(vkCreateImage(device, &ci, nullptr, &s->bundle.colors[i]), "img");
            VkMemoryRequirements req; vkGetImageMemoryRequirements(device, s->bundle.colors[i], &req);
            uint32_t mt = pickMemType(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            ded.image = s->bundle.colors[i];
            VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
            exp.pNext = &ded; exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &exp; ai.allocationSize = req.size; ai.memoryTypeIndex = mt;
            vkok(vkAllocateMemory(device, &ai, nullptr, &s->colorMem[i]), "mem");
            vkok(vkBindImageMemory(device, s->bundle.colors[i], s->colorMem[i], 0), "bind");
        }
        VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dci.imageType = VK_IMAGE_TYPE_2D; dci.format = DEPTH_FMT;
        dci.extent = {extent.width, extent.height, 1};
        dci.mipLevels = 1; dci.arrayLayers = 1; dci.samples = VK_SAMPLE_COUNT_1_BIT;
        dci.tiling = VK_IMAGE_TILING_OPTIMAL;
        dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        dci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        dci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkok(vkCreateImage(device, &dci, nullptr, &s->bundle.depth), "dimg");
        VkMemoryRequirements dreq; vkGetImageMemoryRequirements(device, s->bundle.depth, &dreq);
        uint32_t dmt = pickMemType(phys, dreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dreq.size; dai.memoryTypeIndex = dmt;
        vkok(vkAllocateMemory(device, &dai, nullptr, &s->depthMem), "dmem");
        vkok(vkBindImageMemory(device, s->bundle.depth, s->depthMem, 0), "dbind");
        sc = s;
        return s;
    }
    SwapChainBundle getSwapChainBundle(SwapChainPtr h) override { return ((ExportSwapChain*)h)->bundle; }
    VkResult acquire(SwapChainPtr, ImageSyncData* out) override {
        out->imageIndex = lastAcquired; out->imageReadySemaphore = VK_NULL_HANDLE;
        out->explicitImageReadyWait = nullptr;
        lastAcquired = (lastAcquired + 1) % SC_SIZE; return VK_SUCCESS;
    }
    VkResult present(SwapChainPtr, uint32_t, VkSemaphore) override { return VK_SUCCESS; }
    bool hasResized(SwapChainPtr) override { return false; }
    bool isProtected(SwapChainPtr) override { return false; }
    VkResult recreate(SwapChainPtr) override { return VK_SUCCESS; }
    void destroy(SwapChainPtr h) override {
        auto* s = (ExportSwapChain*)h; VkDevice device = getDevice();
        for (uint32_t i = 0; i < s->bundle.colors.size(); ++i) {
            vkDestroyImage(device, s->bundle.colors[i], nullptr);
            vkFreeMemory(device, s->colorMem[i], nullptr);
        }
        vkDestroyImage(device, s->bundle.depth, nullptr);
        vkFreeMemory(device, s->depthMem, nullptr);
        delete s;
    }
};
}  // namespace

struct Renderer::VulkanState {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue myQueue = VK_NULL_HANDLE;
    uint32_t qfi = 0, filQueueIdx = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdKHR pfnGetMemoryFd = nullptr;
    VkBuffer expBuf = VK_NULL_HANDLE;
    VkDeviceMemory expMem = VK_NULL_HANDLE;
    ExportPlatform platform;
    int cudaDevice = 0;
    cudaExternalMemory_t cudaExt = nullptr;
    void* cudaDptr = nullptr;
};

Renderer::Renderer(const RendererConfig& config)
    : config_(config), vk_(std::make_unique<VulkanState>()) {}

Renderer::~Renderer() { destroy(); }

bool Renderer::initialize() {
    if (initialized_) return true;
    auto& v = *vk_;

    if (!bluevk::initialize()) throw std::runtime_error("bluevk init");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    vkok(vkCreateInstance(&ici, nullptr, &v.instance), "instance");
    bluevk::bindInstance(v.instance);

    uint32_t nd = 0; vkEnumeratePhysicalDevices(v.instance, &nd, nullptr);
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(v.instance, &nd, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID == 0x10DE) { v.phys = d; break; }
    }
    if (!v.phys) throw std::runtime_error("no NVIDIA Vulkan device");

    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(v.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(v.phys, &qn, qfs.data());
    uint32_t qcount = 0;
    for (uint32_t i = 0; i < qn; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { v.qfi = i; qcount = qfs[i].queueCount; break; }
    uint32_t nQ = qcount >= 2 ? 2 : 1;
    v.filQueueIdx = nQ == 2 ? 1 : 0;
    float prio[2] = {1.0f, 1.0f};

    uint32_t en = 0; vkEnumerateDeviceExtensionProperties(v.phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> avail(en);
    vkEnumerateDeviceExtensionProperties(v.phys, nullptr, &en, avail.data());
    const char* wants[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE1_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    std::vector<const char*> enable;
    for (auto* e : wants) if (hasExt(avail, e)) enable.push_back(e);

    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = v.qfi; qci.queueCount = nQ; qci.pQueuePriorities = prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enable.size();
    dci.ppEnabledExtensionNames = enable.data();
    vkok(vkCreateDevice(v.phys, &dci, nullptr, &v.device), "device");
    vkGetDeviceQueue(v.device, v.qfi, 0, &v.myQueue);

    v.pfnGetMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(v.device, "vkGetMemoryFdKHR");
    if (!v.pfnGetMemoryFd) throw std::runtime_error("vkGetMemoryFdKHR unavailable");

    VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &idp; vkGetPhysicalDeviceProperties2(v.phys, &p2);
    int n = 0; cudaGetDeviceCount(&n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp cp; cudaGetDeviceProperties(&cp, i);
        if (memcmp(cp.uuid.bytes, idp.deviceUUID, 16) == 0) { v.cudaDevice = i; break; }
    }
    cudaSetDevice(v.cudaDevice);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = v.qfi;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkok(vkCreateCommandPool(v.device, &pci, nullptr, &v.pool), "pool");

    // Filament on our device + exportable swapchain
    VulkanPlatform::VulkanSharedContext ctx{};
    ctx.instance = v.instance; ctx.physicalDevice = v.phys; ctx.logicalDevice = v.device;
    ctx.graphicsQueueFamilyIndex = v.qfi; ctx.graphicsQueueIndex = v.filQueueIdx;
    engine_ = Engine::Builder()
            .backend(Backend::VULKAN).platform(&v.platform).sharedContext(&ctx)
            .featureLevel(Engine::FeatureLevel::FEATURE_LEVEL_1).build();
    if (!engine_) throw std::runtime_error("Filament engine build failed");

    swapchain_ = engine_->createSwapChain(config_.width, config_.height);
    renderer_ = engine_->createRenderer();
    scene_ = engine_->createScene();
    camera_entity_ = utils::EntityManager::get().create();
    camera_ = engine_->createCamera(camera_entity_);
    camera_->setProjection(70.0,
        double(config_.width) / double(config_.height), 0.05, 200.0);
    camera_->lookAt({3, 3, 3}, {0, 0, 0}, {0, 0, 1});

    setup_view();
    setup_color_grading();

    initialized_ = true;
    return true;
}

void Renderer::setup_view() {
    view_ = engine_->createView();
    view_->setScene(scene_);
    view_->setCamera(camera_);
    view_->setViewport({0, 0, config_.width, config_.height});
    // render into the swapchain (no offscreen RenderTarget) so the swapchain's
    // EXPORTABLE images receive the final post-processed frame.

    filament::Renderer::ClearOptions co;
    co.clearColor = {0.35f, 0.42f, 0.55f, 1.0f};
    co.clear = true;
    renderer_->setClearOptions(co);

    view_->setAntiAliasing(config_.enable_fxaa
        ? View::AntiAliasing::FXAA : View::AntiAliasing::NONE);
    if (config_.enable_msaa)
        view_->setMultiSampleAntiAliasingOptions({.enabled = true, .sampleCount = config_.msaa_samples});
    if (config_.enable_ssao) {
        View::AmbientOcclusionOptions ao;
        ao.radius = 0.3f; ao.power = 1.0f; ao.intensity = 1.0f;
        ao.quality = View::QualityLevel::ULTRA;
        ao.enabled = true; ao.ssct.enabled = true;
        view_->setAmbientOcclusionOptions(ao);
    }
    if (config_.enable_bloom)
        view_->setBloomOptions({.strength = 0.1f, .enabled = true});
    view_->setShadowingEnabled(config_.enable_shadows);
    view_->setDithering(config_.dithering ? View::Dithering::TEMPORAL : View::Dithering::NONE);
}

void Renderer::setup_color_grading() {
    auto b = ColorGrading::Builder();
    b.toneMapping(config_.tone_mapping
        ? ColorGrading::ToneMapping::FILMIC : ColorGrading::ToneMapping::LINEAR);
    b.exposure(config_.exposure);
    color_grading_ = b.build(*engine_);
    view_->setColorGrading(color_grading_);
}

void* Renderer::render_to_cuda() {
    auto& v = *vk_;
    if (renderer_->beginFrame(swapchain_)) {
        renderer_->render(view_);
        renderer_->endFrame();
    }
    engine_->flushAndWait();

    const VkDeviceSize bytes = (VkDeviceSize)config_.width * config_.height * 4;
    uint32_t idx = (v.platform.lastAcquired + SC_SIZE - 1) % SC_SIZE;
    VkImage rendered = v.platform.sc->bundle.colors[idx];

    if (!v.expBuf) {
        VkExternalMemoryBufferCreateInfo extBuf{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        extBuf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.pNext = &extBuf; bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkok(vkCreateBuffer(v.device, &bci, nullptr, &v.expBuf), "expBuf");
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(v.device, v.expBuf, &req);
        uint32_t mt = pickMemType(v.phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        ded.buffer = v.expBuf;
        VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exp.pNext = &ded; exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.pNext = &exp; ai.allocationSize = req.size; ai.memoryTypeIndex = mt;
        vkok(vkAllocateMemory(v.device, &ai, nullptr, &v.expMem), "expMem");
        vkok(vkBindBufferMemory(v.device, v.expBuf, v.expMem, 0), "expBind");
        VkMemoryGetFdInfoKHR gfd{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        gfd.memory = v.expMem; gfd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1; vkok(v.pfnGetMemoryFd(v.device, &gfd, &fd), "fd");
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
        hd.handle.fd = fd; hd.size = req.size; hd.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&v.cudaExt, &hd) != cudaSuccess)
            throw std::runtime_error("cudaImportExternalMemory");
        cudaExternalMemoryBufferDesc bd{}; bd.offset = 0; bd.size = bytes; bd.flags = 0;
        if (cudaExternalMemoryGetMappedBuffer(&v.cudaDptr, v.cudaExt, &bd) != cudaSuccess)
            throw std::runtime_error("cudaExternalMemoryGetMappedBuffer");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = v.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        vkok(vkAllocateCommandBuffers(v.device, &cai, &v.cmd), "cmd");
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(v.cmd, &bi);
    VkImageMemoryBarrier toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toSrc.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.image = rendered;
    toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toSrc.srcAccessMask = 0; toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(v.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSrc);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {config_.width, config_.height, 1};
    vkCmdCopyImageToBuffer(v.cmd, rendered, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           v.expBuf, 1, &region);
    vkEndCommandBuffer(v.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &v.cmd;
    vkQueueSubmit(v.myQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(v.myQueue);
    return v.cudaDptr;
}

int Renderer::cuda_device() const { return vk_->cudaDevice; }

bool Renderer::render_readback_async(uint8_t*) { return false; }  // unused on zero-copy path
void Renderer::finish() { if (engine_) engine_->flushAndWait(); }

void Renderer::destroy() {
    auto& v = *vk_;
    if (v.cudaDptr) { cudaFree(v.cudaDptr); v.cudaDptr = nullptr; }
    if (v.cudaExt) { cudaDestroyExternalMemory(v.cudaExt); v.cudaExt = nullptr; }
    if (engine_) { Engine::destroy(&engine_); engine_ = nullptr; }
    initialized_ = false;
}

}  // namespace vf_mujoco
