#include <CoreFoundation/CoreFoundation.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <cmath>
#include <map>
#include <dlfcn.h> // Needed to load the hidden framework

// Define function pointers for the hidden IOReport tools
typedef struct __IOReportSubscriptionRef *IOReportSubscriptionRef;

typedef CFDictionaryRef (*IOReportCopyChannelsInGroup_t)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*IOReportMergeChannels_t)(CFMutableDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef IOReportSubscriptionRef (*IOReportCreateSubscription_t)(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*IOReportCreateSamples_t)(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*IOReportCreateSamplesDelta_t)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef long (*IOReportSimpleGetIntegerValue_t)(CFDictionaryRef, int);
typedef CFStringRef (*IOReportChannelGetUnitLabel_t)(CFDictionaryRef);

// Global pointers to hold the tools once found
static IOReportCopyChannelsInGroup_t ptr_IOReportCopyChannelsInGroup = nullptr;
static IOReportMergeChannels_t ptr_IOReportMergeChannels = nullptr;
static IOReportCreateSubscription_t ptr_IOReportCreateSubscription = nullptr;
static IOReportCreateSamples_t ptr_IOReportCreateSamples = nullptr;
static IOReportCreateSamplesDelta_t ptr_IOReportCreateSamplesDelta = nullptr;
static IOReportSimpleGetIntegerValue_t ptr_IOReportSimpleGetIntegerValue = nullptr;
static IOReportChannelGetUnitLabel_t ptr_IOReportChannelGetUnitLabel = nullptr;

// Helper to load the framework
void load_ioreport_symbols() {
    if (ptr_IOReportCopyChannelsInGroup) return; // Already loaded

    // Look for the framework in the system directly
    void* handle = dlopen("/System/Library/PrivateFrameworks/IOReport.framework/IOReport", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Warning: Could not load IOReport framework. Power metrics will be 0." << std::endl;
        return;
    }

    // Assign the tools to our pointers
    ptr_IOReportCopyChannelsInGroup = (IOReportCopyChannelsInGroup_t)dlsym(handle, "IOReportCopyChannelsInGroup");
    ptr_IOReportMergeChannels = (IOReportMergeChannels_t)dlsym(handle, "IOReportMergeChannels");
    ptr_IOReportCreateSubscription = (IOReportCreateSubscription_t)dlsym(handle, "IOReportCreateSubscription");
    ptr_IOReportCreateSamples = (IOReportCreateSamples_t)dlsym(handle, "IOReportCreateSamples");
    ptr_IOReportCreateSamplesDelta = (IOReportCreateSamplesDelta_t)dlsym(handle, "IOReportCreateSamplesDelta");
    ptr_IOReportSimpleGetIntegerValue = (IOReportSimpleGetIntegerValue_t)dlsym(handle, "IOReportSimpleGetIntegerValue");
    ptr_IOReportChannelGetUnitLabel = (IOReportChannelGetUnitLabel_t)dlsym(handle, "IOReportChannelGetUnitLabel");
}

std::string cfstring_to_stdstring(CFStringRef cfStr) {
    if (!cfStr) return "";
    const char* c_str = CFStringGetCStringPtr(cfStr, kCFStringEncodingUTF8);
    if (c_str) return std::string(c_str);
    char buffer[128];
    if (CFStringGetCString(cfStr, buffer, sizeof(buffer), kCFStringEncodingUTF8)) {
        return std::string(buffer);
    }
    return "";
}

double convert_to_mj(long raw_value, const std::string& unit) {
    double val = (double)raw_value;
    if (unit == "nJ") return val / 1000000.0;
    if (unit == "uJ" || unit == "µJ") return val / 1000.0;
    if (unit == "mJ") return val;
    if (unit == "J")  return val * 1000.0;
    return val;
}

static void ApplyPowerSample(const void* key, const void* value, void* context) {
    if (!ptr_IOReportSimpleGetIntegerValue || !ptr_IOReportChannelGetUnitLabel) return;

    CFDictionaryRef sample = (CFDictionaryRef)value;
    double* total_energy_mj = (double*)context;

    long raw_value = ptr_IOReportSimpleGetIntegerValue(sample, 0);
    CFStringRef unitRef = ptr_IOReportChannelGetUnitLabel(sample);
    std::string unit = cfstring_to_stdstring(unitRef);

    *total_energy_mj += convert_to_mj(raw_value, unit);
}

class Monitor {
private:
    IOReportSubscriptionRef subscription = nullptr;
    CFMutableDictionaryRef channels = nullptr;

public:
    Monitor() {
        // Load the tools first
        load_ioreport_symbols();

        if (!ptr_IOReportCreateSubscription || !ptr_IOReportCopyChannelsInGroup) {
            return;
        }

        channels = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        CFStringRef energyGroup = CFStringCreateWithCString(kCFAllocatorDefault, "Energy Model", kCFStringEncodingUTF8);

        // Use the loaded pointer
        CFDictionaryRef energyChannels = ptr_IOReportCopyChannelsInGroup(energyGroup, NULL, 0, 0, 0);

        if (energyChannels) {
            ptr_IOReportMergeChannels(channels, energyChannels, NULL);
            CFRelease(energyChannels);
        }
        CFRelease(energyGroup);

        subscription = ptr_IOReportCreateSubscription(NULL, channels, NULL, 0, NULL);
    }

    ~Monitor() {
        if (subscription) CFRelease(subscription);
        if (channels) CFRelease(channels);
    }

    double get_current_power() {
        if (!subscription || !ptr_IOReportCreateSamples) return 0.0;

        CFDictionaryRef sample1 = ptr_IOReportCreateSamples(subscription, channels, NULL);
        if (!sample1) return 0.0;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        CFDictionaryRef sample2 = ptr_IOReportCreateSamples(subscription, channels, NULL);
        if (!sample2) {
            CFRelease(sample1);
            return 0.0;
        }

        CFDictionaryRef delta = ptr_IOReportCreateSamplesDelta(sample1, sample2, NULL);

        double total_energy_mj = 0.0;
        CFDictionaryApplyFunction(delta, ApplyPowerSample, &total_energy_mj);

        CFRelease(sample1);
        CFRelease(sample2);
        CFRelease(delta);

        return total_energy_mj / 100.0;
    }

    std::string get_gpu_utilization() {
        return "Not Implemented Yet";
    }
};