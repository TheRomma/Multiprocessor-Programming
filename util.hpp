#pragma once

#include <cinttypes>

void sqMatrixProduct(
	const float* a,
	const float* b,
	const uint32_t side,
	float* out
);

void printSqMatrix(
	const float* a,
	const uint32_t side
);

void imgLoad(
	const char* filename,
	uint32_t* width,
	uint32_t* height,
	unsigned char** image
);

void imgWrite(
	const char* filename,
	const uint32_t width,
	const uint32_t height,
	const unsigned char* image
);
