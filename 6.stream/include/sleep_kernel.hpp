#ifndef SLEEP_KERNEL
#define SLEEP_KERNEL

void SleepSingleStream(float *src_host, int width, int block_size, int cout);
void SleepMultiStream(float *src_host, int width, int block_size, int cout);


#endif // SLEEP_KERNEL