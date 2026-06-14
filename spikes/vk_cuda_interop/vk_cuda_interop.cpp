// Phase 2 isolation spike: Vulkan <-> CUDA zero-copy external memory.
//
// Proves the core mechanism Option B depends on:
//   Vulkan allocates DEVICE_LOCAL memory, EXPORTS it as an opaque FD, writes a
//   known pattern into it (vkCmdFillBuffer), and CUDA IMPORTS the same FD and
//   reads the bytes back -- WITHOUT any host copy between the two APIs.
//
// This is the exact direction we need for mujofil-warp: Filament (Vulkan)
// produces pixels in VRAM, PyTorch (CUDA) consumes them with no readback.
//
// No Filament, no MuJoCo, no CUDA kernel -- just the interop primitive.
//
// Build: see build.sh.   Run: ./vk_cuda_interop
// Exit 0 + "INTEROP OK" => zero-copy path is viable on this driver.

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult err = (x);                                                    \
        if (err != VK_SUCCESS) {                                               \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", err, __FILE__,       \
                    __LINE__);                                                 \
            return 2;                                                          \
        }                                                                      \
    } while (0)

#define CU_CHECK(x)                                                            \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            return 3;                                                          \
        }                                                                      \
    } while (0)

static const uint32_t WIDTH = 256, HEIGHT = 256;
static const VkDeviceSize PAYLOAD = (VkDeviceSize)WIDTH * HEIGHT * 4; // RGBA8
static const uint32_t FILL_WORD = 0xABCDEF01u; // pattern Vulkan writes

int main() {
    // ---- 1. Vulkan instance (1.1 for external-memory capabilities) ----------
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vk_cuda_interop";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;

    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    // ---- 2. Pick the NVIDIA discrete GPU (skip AMD iGPU / llvmpipe) ----------
    uint32_t ndev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &ndev, nullptr));
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &ndev, devs.data()));

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    uint8_t vkUUID[VK_UUID_SIZE] = {0};
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID == 0x10DE /* NVIDIA */) {
            phys = d;
            props = p;
            VkPhysicalDeviceIDProperties idp{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &idp;
            vkGetPhysicalDeviceProperties2(d, &p2);
            memcpy(vkUUID, idp.deviceUUID, VK_UUID_SIZE);
            break;
        }
    }
    if (phys == VK_NULL_HANDLE) {
        fprintf(stderr, "No NVIDIA Vulkan device found\n");
        return 1;
    }
    printf("Vulkan device: %s\n", props.deviceName);

    // ---- 3. Match the SAME GPU on the CUDA side via UUID --------------------
    int cudaCount = 0;
    CU_CHECK(cudaGetDeviceCount(&cudaCount));
    int cudaDev = -1;
    for (int i = 0; i < cudaCount; ++i) {
        cudaDeviceProp cp;
        CU_CHECK(cudaGetDeviceProperties(&cp, i));
        if (memcmp(cp.uuid.bytes, vkUUID, 16) == 0) {
            cudaDev = i;
            printf("CUDA device:   %s (UUID matches Vulkan)\n", cp.name);
            break;
        }
    }
    if (cudaDev < 0) {
        fprintf(stderr, "No CUDA device matches the Vulkan device UUID\n");
        return 1;
    }
    CU_CHECK(cudaSetDevice(cudaDev));

    // ---- 4. Logical device with external-memory-fd ext ----------------------
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfs.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    const char* devExts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = devExts;

    VkDevice device;
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, qfi, 0, &queue);

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(
        device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        fprintf(stderr, "vkGetMemoryFdKHR not available\n");
        return 1;
    }

    // ---- 5. Exportable buffer (DEVICE_LOCAL), dedicated allocation -----------
    VkExternalMemoryBufferCreateInfo extBuf{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    extBuf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &extBuf;
    bci.size = PAYLOAD;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buffer));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    uint32_t memType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memType = i;
            break;
        }
    }
    if (memType == UINT32_MAX) { fprintf(stderr, "no device-local mem\n"); return 1; }

    VkMemoryDedicatedAllocateInfo dedicated{
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer;

    VkExportMemoryAllocateInfo expInfo{
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    expInfo.pNext = &dedicated;
    expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &expInfo;
    mai.allocationSize = memReq.size;
    mai.memoryTypeIndex = memType;

    VkDeviceMemory memory;
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    // ---- 6. Vulkan WRITES the pattern into VRAM (vkCmdFillBuffer) ------------
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = qfi;
    VkCommandPool pool;
    VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));

    VkCommandBufferAllocateInfo cbai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdFillBuffer(cmd, buffer, 0, PAYLOAD, FILL_WORD);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue)); // coarse sync (spike); real path: semaphore

    // ---- 7. Export the memory as an opaque FD -------------------------------
    VkMemoryGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    VK_CHECK(vkGetMemoryFdKHR(device, &fdInfo, &fd));
    printf("Exported Vulkan memory as FD %d (%llu bytes)\n", fd,
           (unsigned long long)memReq.size);

    // ---- 8. CUDA IMPORTS the same FD (CUDA takes ownership of fd) ------------
    cudaExternalMemory_t extMem;
    cudaExternalMemoryHandleDesc hd{};
    hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = fd;
    hd.size = memReq.size;
    hd.flags = cudaExternalMemoryDedicated; // matched the dedicated alloc above
    CU_CHECK(cudaImportExternalMemory(&extMem, &hd));

    void* dptr = nullptr;
    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = PAYLOAD;
    bd.flags = 0;
    CU_CHECK(cudaExternalMemoryGetMappedBuffer(&dptr, extMem, &bd));
    printf("CUDA mapped the imported memory at device ptr %p\n", dptr);

    // ---- 9. CUDA reads it back; verify it sees Vulkan's pattern -------------
    std::vector<uint32_t> host(PAYLOAD / 4, 0);
    CU_CHECK(cudaMemcpy(host.data(), dptr, PAYLOAD, cudaMemcpyDeviceToHost));
    CU_CHECK(cudaDeviceSynchronize());

    size_t mismatches = 0;
    for (uint32_t v : host)
        if (v != FILL_WORD) ++mismatches;

    printf("Verify: %zu / %zu words == 0x%08X (Vulkan-written)\n",
           host.size() - mismatches, host.size(), FILL_WORD);

    // ---- cleanup ------------------------------------------------------------
    cudaFree(dptr);
    cudaDestroyExternalMemory(extMem);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    if (mismatches == 0) {
        printf("\nINTEROP OK: Vulkan-produced VRAM read zero-copy by CUDA.\n");
        printf("=> Option B zero-copy pixel path is VIABLE on this driver.\n");
        return 0;
    }
    fprintf(stderr, "\nINTEROP FAILED: %zu mismatching words.\n", mismatches);
    return 1;
}
