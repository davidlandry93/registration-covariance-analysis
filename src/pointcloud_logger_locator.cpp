
#include "null_pointcloud_logger.h"
#include "pointcloud_logger_locator.h"

namespace recova {

std::unique_ptr<PointcloudLogger> PointcloudLoggerLocator::logger(new NullPointcloudLogger);

PointcloudLoggerLocator::PointcloudLoggerLocator() {
    if(!logger) {
        logger.reset(new NullPointcloudLogger);
    }
}

PointcloudLogger& PointcloudLoggerLocator::get() {
    return *logger;
}

void PointcloudLoggerLocator::set(std::unique_ptr<PointcloudLogger>&& new_logger) {
    logger = std::move(new_logger);
}

}
