// Wuji Hand2 CRAWL: C side of the mjwarp hybrid. The mechanism (embedded
// Python bootstrap, [env] kwarg forwarding, captured-graph launch, log/render
// bridges) lives in src/mjwarp_host.cuh; the task itself is in
// wujicrawl_warp.py.
#define MJWARP_ENV_NAME "wujicrawl"
#define MJWARP_PY_ENVVAR "WUJICRAWL_PY"
#include "mjwarp_host.cuh"
