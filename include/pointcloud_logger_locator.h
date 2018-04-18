#ifndef RECOVA_POINTCLOUD_LOGGER_LOCATOR
#define RECOVA_POINTCLOUD_LOGGER_LOCATOR

#include <memory>

#include "pointcloud_logger.h"

namespace recova {
class PointcloudLoggerLocator {
  public:
    PointcloudLoggerLocator();
    static PointcloudLogger& get();
    static void set(std::unique_ptr<PointcloudLogger>&& new_logger);
  private:
    static std::unique_ptr<PointcloudLogger> logger;
};
}

#endif
