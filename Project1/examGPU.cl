void kernel doConv(	global const int* intens,
					global int* filtred,
					global const int* kern,
					const int rows,
					const int cols)
{
	int col = get_global_id(0);
	int row = get_global_id(1);
	for (int r_ = -1; r_ <= 1; r_++) {
		for (int c_ = -1; c_ <= 1; c_++) {
			filtred[row * cols + col] += intens[(row + r_ + 1) * (cols + 2) + col + c_ + 1] * kern[(r_ + 1) * 3 + (c_ + 1)];
		}
	}
}