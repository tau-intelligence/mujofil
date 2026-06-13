// Phase 2, milestone 2: shared-context zero-copy WITHOUT rebuilding Filament.
//
// Milestone 1 showed Filament's DEFAULT device lacks VK_KHR_external_memory_fd.
// The fix that avoids a from-source rebuild: WE create the VkInstance/Device
// (with the external-memory extensions enabled) and hand them to Filament via
// VulkanSharedContext + Engine::Builder::sharedContext(). Filament then renders
// on OUR device, on which we CAN allocate exportable images for CUDA.
//
// This spike proves:
//   1. We can create a Vulkan device (via bluevk) with external-memory-fd +
//      the device extensions Filament needs (swapchain/maintenance/multiview).
//   2. Filament accepts it as a shared context and builds an Engine.
//   3. vkGetMemoryFdKHR now RESOLVES on that device.
//   4. We can allocate an EXPORTABLE image on it and export an opaque FD.
// (CUDA import of such an FD was already proven in spikes/vk_cuda_interop.)
//
// Build: cmake -S . -B build && cmake --build build
// Run:   ./build/filament_vk_export
// Exit 0 + "PHASE2 SHARED-CONTEXT OK" => zero-copy path unblocked, no rebuild.

// Do NOT include <vulkan/vulkan.h> directly — BlueVK provides the Vk* symbols.
#include <filament/Engine.h>
#include <backend/platforms/VulkanPlatform.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace filament;
using namespace filament::backend;
using namespace bluevk;

#define VK_OK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "Vk error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    return 2; } } while (0)

static bool hasExt(const std::vector<VkExtensionProperties>& a, const char* n) {
    for (auto& e : a) if (!strcmp(e.extensionName, n)) return true;
    return false;
}

int main() {
    if (!bluevk::initialize()) {
        fprintf(stderr, "FAILED: bluevk::initialize()\n");
        return 1;
    }

    // ---- 1. Instance (api 1.3; external-memory caps are core in 1.1+) --------
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "filament_vk_export";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    VK_OK(vkCreateInstance(&ici, nullptr, &instance));
    bluevk::bindInstance(instance);

    // ---- 2. Pick the NVIDIA physical device ---------------------------------
    uint32_t nd = 0;
    VK_OK(vkEnumeratePhysicalDevices(instance, &nd, nullptr));
    std::vector<VkPhysicalDevice> devs(nd);
    VK_OK(vkEnumeratePhysicalDevices(instance, &nd, devs.data()));
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID == 0x10DE) { phys = d; props = p; break; }
    }
    if (!phys) { fprintf(stderr, "no NVIDIA device\n"); return 1; }
    printf("Physical device: %s\n", props.deviceName);

    // ---- 3. Graphics queue family; request 2 queues (one for us, one for FL) -
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qfs.data());
    uint32_t qfi = UINT32_MAX, qcount = 0;
    for (uint32_t i = 0; i < qn; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            qfi = i; qcount = qfs[i].queueCount; break;
        }
    if (qfi == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 1; }
    uint32_t nQueues = qcount >= 2 ? 2 : 1;   // index 0 = ours, 1 = Filament
    uint32_t filamentQueueIndex = nQueues == 2 ? 1 : 0;
    float prio[2] = {1.0f, 1.0f};

    // ---- 4. Device extensions: Filament's + external-memory-fd --------------
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> avail(en);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, avail.data());

    std::vector<const char*> want = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,          // Filament
        VK_KHR_MAINTENANCE1_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME,
        VK_KHR_MAINTENANCE3_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,    // ours (zero-copy)
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    std::vector<const char*> enable;
    for (auto* e : want) if (hasExt(avail, e)) enable.push_back(e);
    printf("Enabling %zu/%zu device extensions (incl external-memory-fd=%d)\n",
           enable.size(), want.size(),
           (int)hasExt(avail, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME));

    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = nQueues;
    qci.pQueuePriorities = prio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enable.size();
    dci.ppEnabledExtensionNames = enable.data();

    VkDevice device;
    VK_OK(vkCreateDevice(phys, &dci, nullptr, &device));

    // ---- 5. Hand our device to Filament via shared context ------------------
    VulkanPlatform::VulkanSharedContext ctx{};
    ctx.instance = instance;
    ctx.physicalDevice = phys;
    ctx.logicalDevice = device;
    ctx.graphicsQueueFamilyIndex = qfi;
    ctx.graphicsQueueIndex = filamentQueueIndex;

    Engine* engine = Engine::Builder()
            .backend(Backend::VULKAN)
            .sharedContext(&ctx)
            .featureLevel(Engine::FeatureLevel::FEATURE_LEVEL_1)
            .build();
    if (!engine) {
        fprintf(stderr, "FAILED: Engine build on shared context\n");
        return 1;
    }
    printf("Filament Engine built on OUR shared device.\n");

    // ---- 6. Is external-memory-fd now live on the device Filament uses? ------
    auto pfnGetMemoryFd =
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    printf("vkGetMemoryFdKHR on shared device: %s\n",
           pfnGetMemoryFd ? "RESOLVED" : "NULL");
    if (!pfnGetMemoryFd) { Engine::destroy(&engine); return 2; }

    // ---- 7. Allocate an EXPORTABLE image on the shared device + export FD ----
    const uint32_t W = 256, H = 256;
    VkExternalMemoryImageCreateInfo extImg{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    extImg.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo ici2{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici2.pNext = &extImg;
    ici2.imageType = VK_IMAGE_TYPE_2D;
    ici2.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici2.extent = {W, H, 1};
    ici2.mipLevels = 1;
    ici2.arrayLayers = 1;
    ici2.samples = VK_SAMPLE_COUNT_1_BIT;
    ici2.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici2.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici2.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici2.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    VK_OK(vkCreateImage(device, &ici2, nullptr, &image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, image, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t memType = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memType = i; break;
        }
    if (memType == UINT32_MAX) { fprintf(stderr, "no device-local mem\n"); return 1; }

    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.image = image;
    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exp.pNext = &ded;
    exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &exp;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memType;
    VkDeviceMemory memory;
    VK_OK(vkAllocateMemory(device, &mai, nullptr, &memory));
    VK_OK(vkBindImageMemory(device, image, memory, 0));

    VkMemoryGetFdInfoKHR gfd{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gfd.memory = memory;
    gfd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    VK_OK(pfnGetMemoryFd(device, &gfd, &fd));
    printf("Allocated exportable %ux%u image; exported FD %d (%llu bytes)\n",
           W, H, fd, (unsigned long long)req.size);

    printf("\nPHASE2 SHARED-CONTEXT OK: Filament runs on our device, which can "
           "export image memory to CUDA. Zero-copy path is unblocked WITHOUT a "
           "from-source Filament rebuild.\n");

    vkDestroyImage(device, image, nullptr);
    vkFreeMemory(device, memory, nullptr);
    Engine::destroy(&engine);
    return 0;
}
