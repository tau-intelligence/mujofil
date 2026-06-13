// Phase 2, milestone 3a: Filament renders into OUR exportable images.
//
// Builds on M2 (shared device with external-memory-fd). Here we subclass
// VulkanPlatform to provide a HEADLESS swapchain whose color images are OUR
// EXPORTABLE images. Filament renders a frame (clearing to a known color) into
// one of them. We then blit that image to a host buffer and verify the color —
// proving Filament rendered into an image we own and can export.
//
// (3b will replace the CPU verification with FD -> CUDA -> torch.)
//
// Headless swapchains need NO acquire/present semaphores (per Filament's own
// VulkanPlatformHeadlessSwapChain): acquire just returns a round-robin index,
// present is a no-op. So our override is simple — only the images differ.

#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/Viewport.h>
#include <filament/SwapChain.h>
#include <backend/platforms/VulkanPlatform.h>
#include <utils/EntityManager.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace filament;
using namespace filament::backend;
using namespace bluevk;

#define VK_OK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "Vk error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    std::exit(2); } } while (0)

static constexpr uint32_t SC_SIZE = 2;           // match Filament headless
static constexpr VkFormat COLOR_FMT = VK_FORMAT_R8G8B8A8_UNORM;
static constexpr VkFormat DEPTH_FMT = VK_FORMAT_D32_SFLOAT;

static uint32_t pickMemType(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

// ----- Our exportable headless swapchain ------------------------------------
struct ExportSwapChain : public Platform::SwapChain {
    VulkanPlatform::SwapChainBundle bundle;
    std::vector<VkDeviceMemory> colorMem;
    VkDeviceMemory depthMem = VK_NULL_HANDLE;
    std::vector<int> exportedFd;   // one per color image
};

class ExportPlatform : public VulkanPlatform {
public:
    ExportSwapChain* sc = nullptr;
    uint32_t lastAcquired = 0;

    // Don't set a GPU preference: illegal together with a shared context.
    Customization getCustomization() const noexcept override {
        Customization c;
        // Leave the image in PRESENT_SRC after the frame so we can blit it.
        c.transitionSwapChainImageLayoutForPresent = true;
        return c;
    }

    SwapChainPtr createSwapChain(void* nativeWindow, uint64_t flags,
                                 VkExtent2D extent) override {
        VkDevice device = getDevice();
        VkPhysicalDevice phys = getPhysicalDevice();

        auto* s = new ExportSwapChain();
        s->bundle.extent = extent;
        s->bundle.colorFormat = COLOR_FMT;
        s->bundle.depthFormat = DEPTH_FMT;
        s->bundle.colors.reserve(SC_SIZE);
        s->bundle.colors.resize(SC_SIZE);
        s->colorMem.resize(SC_SIZE);
        s->exportedFd.resize(SC_SIZE, -1);

        for (uint32_t i = 0; i < SC_SIZE; ++i) {
            // exportable color image
            VkExternalMemoryImageCreateInfo ext{
                VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
            ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

            VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ci.pNext = &ext;
            ci.imageType = VK_IMAGE_TYPE_2D;
            ci.format = COLOR_FMT;
            ci.extent = {extent.width, extent.height, 1};
            ci.mipLevels = 1;
            ci.arrayLayers = 1;
            ci.samples = VK_SAMPLE_COUNT_1_BIT;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL;
            ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VK_OK(vkCreateImage(device, &ci, nullptr, &s->bundle.colors[i]));

            VkMemoryRequirements req;
            vkGetImageMemoryRequirements(device, s->bundle.colors[i], &req);
            uint32_t mt = pickMemType(phys, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            VkMemoryDedicatedAllocateInfo ded{
                VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            ded.image = s->bundle.colors[i];
            VkExportMemoryAllocateInfo exp{
                VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
            exp.pNext = &ded;
            exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext = &exp;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            VK_OK(vkAllocateMemory(device, &ai, nullptr, &s->colorMem[i]));
            VK_OK(vkBindImageMemory(device, s->bundle.colors[i], s->colorMem[i], 0));
        }

        // depth (not exportable)
        VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dci.imageType = VK_IMAGE_TYPE_2D;
        dci.format = DEPTH_FMT;
        dci.extent = {extent.width, extent.height, 1};
        dci.mipLevels = 1;
        dci.arrayLayers = 1;
        dci.samples = VK_SAMPLE_COUNT_1_BIT;
        dci.tiling = VK_IMAGE_TILING_OPTIMAL;
        dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        dci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        dci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_OK(vkCreateImage(device, &dci, nullptr, &s->bundle.depth));
        VkMemoryRequirements dreq;
        vkGetImageMemoryRequirements(device, s->bundle.depth, &dreq);
        uint32_t dmt = pickMemType(phys, dreq.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dreq.size;
        dai.memoryTypeIndex = dmt;
        VK_OK(vkAllocateMemory(device, &dai, nullptr, &s->depthMem));
        VK_OK(vkBindImageMemory(device, s->bundle.depth, s->depthMem, 0));

        sc = s;
        printf("createSwapChain: %u exportable %ux%u color images + depth\n",
               SC_SIZE, extent.width, extent.height);
        return s;
    }

    SwapChainBundle getSwapChainBundle(SwapChainPtr handle) override {
        return ((ExportSwapChain*)handle)->bundle;
    }

    VkResult acquire(SwapChainPtr handle, ImageSyncData* out) override {
        out->imageIndex = lastAcquired;
        out->imageReadySemaphore = VK_NULL_HANDLE;   // headless: no sync needed
        out->explicitImageReadyWait = nullptr;
        lastAcquired = (lastAcquired + 1) % SC_SIZE;
        return VK_SUCCESS;
    }

    VkResult present(SwapChainPtr, uint32_t, VkSemaphore) override {
        return VK_SUCCESS;  // headless no-op
    }

    // The base versions of these look up an internal swapchain registry that our
    // custom handle isn't in (-> "Bad handle" panic). Override with fixed values.
    bool hasResized(SwapChainPtr) override { return false; }
    bool isProtected(SwapChainPtr) override { return false; }
    VkResult recreate(SwapChainPtr) override { return VK_SUCCESS; }

    void destroy(SwapChainPtr handle) override {
        auto* s = (ExportSwapChain*)handle;
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

// ----- device creation (same as M2) -----------------------------------------
static bool hasExt(const std::vector<VkExtensionProperties>& a, const char* n) {
    for (auto& e : a) if (!strcmp(e.extensionName, n)) return true;
    return false;
}

int main() {
    if (!bluevk::initialize()) { fprintf(stderr, "bluevk init failed\n"); return 1; }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    VK_OK(vkCreateInstance(&ici, nullptr, &instance));
    bluevk::bindInstance(instance);

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(instance, &nd, nullptr);
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(instance, &nd, devs.data());
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID == 0x10DE) { phys = d; break; }
    }
    if (!phys) { fprintf(stderr, "no NVIDIA device\n"); return 1; }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qfs.data());
    uint32_t qfi = UINT32_MAX, qcount = 0;
    for (uint32_t i = 0; i < qn; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfi = i; qcount = qfs[i].queueCount; break; }
    uint32_t nQueues = qcount >= 2 ? 2 : 1;
    uint32_t filQueueIdx = nQueues == 2 ? 1 : 0;   // ours = 0, Filament = 1
    float prio[2] = {1.0f, 1.0f};

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> avail(en);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, avail.data());
    std::vector<const char*> want = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_MAINTENANCE1_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    std::vector<const char*> enable;
    for (auto* e : want) if (hasExt(avail, e)) enable.push_back(e);

    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = nQueues; qci.pQueuePriorities = prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enable.size();
    dci.ppEnabledExtensionNames = enable.data();
    VkDevice device;
    VK_OK(vkCreateDevice(phys, &dci, nullptr, &device));
    VkQueue myQueue;
    vkGetDeviceQueue(device, qfi, 0, &myQueue);   // ours (index 0)

    // ----- Filament on our device + our exportable-swapchain platform -------
    ExportPlatform platform;
    VulkanPlatform::VulkanSharedContext ctx{};
    ctx.instance = instance; ctx.physicalDevice = phys; ctx.logicalDevice = device;
    ctx.graphicsQueueFamilyIndex = qfi; ctx.graphicsQueueIndex = filQueueIdx;

    Engine* engine = Engine::Builder()
            .backend(Backend::VULKAN).platform(&platform).sharedContext(&ctx)
            .featureLevel(Engine::FeatureLevel::FEATURE_LEVEL_1).build();
    if (!engine) { fprintf(stderr, "engine build failed\n"); return 1; }

    const uint32_t W = 256, H = 256;
    filament::SwapChain* swapChain = engine->createSwapChain(W, H);
    Renderer* renderer = engine->createRenderer();
    Scene* scene = engine->createScene();
    View* view = engine->createView();
    utils::Entity camEnt = utils::EntityManager::get().create();
    Camera* camera = engine->createCamera(camEnt);
    view->setScene(scene);
    view->setCamera(camera);
    view->setViewport({0, 0, W, H});
    view->setPostProcessingEnabled(false);

    const math::float4 CLEAR = {0.20f, 0.60f, 0.90f, 1.0f};  // distinctive
    Renderer::ClearOptions co;
    co.clearColor = CLEAR;
    co.clear = true;
    renderer->setClearOptions(co);

    // render one frame into our exportable image
    if (renderer->beginFrame(swapChain)) {
        renderer->render(view);
        renderer->endFrame();
    }
    engine->flushAndWait();
    printf("Filament rendered one frame (clear=%.2f,%.2f,%.2f).\n",
           CLEAR.x, CLEAR.y, CLEAR.z);

    // ----- blit the rendered image to a host buffer to verify ---------------
    // Filament rendered into colors[lastAcquired-1] (acquire incremented after).
    uint32_t idx = (platform.lastAcquired + SC_SIZE - 1) % SC_SIZE;
    VkImage rendered = platform.sc->bundle.colors[idx];

    VkDeviceSize bytes = (VkDeviceSize)W * H * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer hostBuf; VK_OK(vkCreateBuffer(device, &bci, nullptr, &hostBuf));
    VkMemoryRequirements breq; vkGetBufferMemoryRequirements(device, hostBuf, &breq);
    uint32_t hmt = pickMemType(phys, breq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = breq.size; bai.memoryTypeIndex = hmt;
    VkDeviceMemory hostMem; VK_OK(vkAllocateMemory(device, &bai, nullptr, &hostMem));
    VK_OK(vkBindBufferMemory(device, hostBuf, hostMem, 0));

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = qfi;
    VkCommandPool pool; VK_OK(vkCreateCommandPool(device, &pci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; VK_OK(vkAllocateCommandBuffers(device, &cai, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_OK(vkBeginCommandBuffer(cmd, &bi));

    // transition rendered image PRESENT_SRC -> TRANSFER_SRC (preserve contents)
    VkImageMemoryBarrier toSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toSrc.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.image = rendered;
    toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toSrc.srcAccessMask = 0;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSrc);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, rendered, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           hostBuf, 1, &region);
    VK_OK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VK_OK(vkQueueSubmit(myQueue, 1, &si, VK_NULL_HANDLE));
    VK_OK(vkQueueWaitIdle(myQueue));

    uint8_t* px = nullptr;
    VK_OK(vkMapMemory(device, hostMem, 0, bytes, 0, (void**)&px));
    // sample the center pixel
    uint32_t cx = W / 2, cy = H / 2;
    uint8_t* p = px + (cy * W + cx) * 4;
    uint8_t er = (uint8_t)(CLEAR.x * 255 + 0.5f);
    uint8_t eg = (uint8_t)(CLEAR.y * 255 + 0.5f);
    uint8_t eb = (uint8_t)(CLEAR.z * 255 + 0.5f);
    printf("center pixel = (%u,%u,%u,%u), expected ~(%u,%u,%u)\n",
           p[0], p[1], p[2], p[3], er, eg, eb);
    int dr = abs((int)p[0] - er), dg = abs((int)p[1] - eg), db = abs((int)p[2] - eb);
    bool ok = dr <= 4 && dg <= 4 && db <= 4;
    vkUnmapMemory(device, hostMem);

    if (ok) {
        printf("\nRENDER-INTO-EXPORTABLE OK: Filament rendered the clear color "
               "into our exportable image. (3b: route this image to CUDA.)\n");
    } else {
        printf("\nMISMATCH: rendered pixels != clear color. Layout/format issue.\n");
    }

    Engine::destroy(&engine);
    return ok ? 0 : 1;
}
