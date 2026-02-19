#include <CoreFoundation/CoreFoundation.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <cmath>
#include <map>

extern "C" {
    typedef struct __IOReportSubscriptionRef *IOReportSubscriptionRef;

    // getting the dictionary of the available channels
    CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef group, CFStringRef subGroup, uint64_t options, uint64_t graphOptions, uint64_t debugOptions);
    void IOReportMergeChannels(CFMutableDictionaryRef destination, CFDictionaryRef source, CFTypeRef reserved);

    // sub to the list of channels
    IOReportSubscriptionRef IOReportCreateSubscription(void* a, CFMutableDictionaryRef channels, CFMutableDictionaryRef* b, uint64_t c, CFTypeRef d);

    // snapshot of the current values
    CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef sub, CFMutableDictionaryRef channels, CFTypeRef reserved);
    CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef prev, CFDictionaryRef current, CFTypeRef reserved);

    // Helpers: Get raw integer
    long IOReportSimpleGetIntegerValue(CFDictionaryRef sample, int index);
    // unit label
    CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef sample);

}

std::string cfstring_to_stdstring(CFStringRef cfStr) {
    if (!cfStr) return "";
    const char* c_str = CFStringGetCStringPtr(cfStr, kCFStringEncodingUTF8);
    if (c_str) return std::string(c_str);

    // fallback
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
    // 1. Cast the generic pointers
    // The 'value' in the dictionary is the Sample (CFDictionaryRef)
    CFDictionaryRef sample = (CFDictionaryRef)value;
    double* total_energy_mj = (double*)context;

    // 2. Get the raw integer
    long raw_value = IOReportSimpleGetIntegerValue(sample, 0);

    // 3. Get the unit
    CFStringRef unitRef = IOReportChannelGetUnitLabel(sample);
    std::string unit = cfstring_to_stdstring(unitRef);

    // 4. Add to total (using our helper)
    *total_energy_mj += convert_to_mj(raw_value, unit);
}

class Monitor {
private:
    IOReportSubscriptionRef subscription = nullptr;
    CFMutableDictionaryRef channels = nullptr;

public:
    Monitor() {
        channels = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

        CFStringRef energyGroup = CFStringCreateWithCString(kCFAllocatorDefault, "Energy Model", kCFStringEncodingUTF8);
        CFDictionaryRef energyChannels = IOReportCopyChannelsInGroup(energyGroup, NULL, 0, 0, 0);

        if (energyChannels) {
            IOReportMergeChannels(channels, energyChannels, NULL);
            CFRelease(energyChannels);
        }
        CFRelease(energyGroup);

        subscription = IOReportCreateSubscription(NULL, channels, NULL, 0, NULL);
    }

    ~Monitor() {
        if (subscription) CFRelease(subscription);
        if (channels) CFRelease(channels);
    }

    double get_current_power() {
        if (!subscription) return 0.0;

        CFDictionaryRef sample1 = IOReportCreateSamples(subscription, channels, NULL);
        if (!sample1) return 0.0;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        CFDictionaryRef sample2 = IOReportCreateSamples(subscription, channels, NULL);
        if (!sample2) {
            CFRelease(sample1);
            return 0.0;
        }

        CFDictionaryRef delta = IOReportCreateSamplesDelta(sample1, sample2, NULL);

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