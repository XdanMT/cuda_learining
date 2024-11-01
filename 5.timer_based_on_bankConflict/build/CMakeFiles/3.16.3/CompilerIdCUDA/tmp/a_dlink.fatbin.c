#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000350,0x0000004001010002,0x0000000000000310\n"
".quad 0x0000000000000000,0x0000003400010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007400be0002,0x0000000000000000,0x0000000000000000,0x00000000000001d0\n"
".quad 0x0000004000340534,0x0001000500400000,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x65722e766e2e006f,0x6e6f697463612e6c,0x72747368732e0000,0x7274732e00626174\n"
".quad 0x6d79732e00626174,0x6d79732e00626174,0x646e68735f626174,0x6e692e766e2e0078\n"
".quad 0x722e766e2e006f66,0x6f697463612e6c65,0x000000000000006e,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0004000300000032,0x0000000000000000\n"
".quad 0x0000000000000000,0x000000000000004b,0x222f0a1008020200,0x0000000008000000\n"
".quad 0x0000000008080000,0x0000000008100000,0x0000000008180000,0x0000000008200000\n"
".quad 0x0000000008280000,0x0000000008300000,0x0000000008380000,0x0000000008000001\n"
".quad 0x0000000008080001,0x0000000008100001,0x0000000008180001,0x0000000008200001\n"
".quad 0x0000000008280001,0x0000000008300001,0x0000000008380001,0x0000000008000002\n"
".quad 0x0000000008080002,0x0000000008100002,0x0000000008180002,0x0000000008200002\n"
".quad 0x0000000008280002,0x0000000008300002,0x0000000008380002,0x0000002c14000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000040\n"
".quad 0x0000000000000041,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x0000000000000081\n"
".quad 0x0000000000000041,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x00000000000000c8\n"
".quad 0x0000000000000030,0x0000000200000002,0x0000000000000008,0x0000000000000018\n"
".quad 0x7000000b00000032,0x0000000000000000,0x0000000000000000,0x00000000000000f8\n"
".quad 0x00000000000000d8,0x0000000000000000,0x0000000000000008,0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[108];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 2, fatbinData, (void**)__cudaPrelinkedFatbins };
#ifdef __cplusplus
}
#endif
