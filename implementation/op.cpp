#include <torch/script.h>

uint16_t leadingBitPosition(uint16_t val)
{
	uint16_t clz;
	// clz function calculates number of leading zeros in integer number
	clz = __builtin_clz(val);
	return 31-clz;
}

int DRALM( int a,  int b, unsigned short w) {
    unsigned short n;
	n = 16;
	if(a == 0 || b == 0) return 0;
	char sgn_a = a > 0 ? 0 : 1;
	char sgn_b = b > 0 ? 0 : 1;
	unsigned int a_abs = sgn_a ? -(a)-1  : a;
	unsigned int b_abs = sgn_b ? -(b)-1 : b;

	unsigned int k_a, x_a;
	k_a = leadingBitPosition(a_abs);
	x_a = a_abs << (n - 1 - k_a);
    //printf("Xa = %x \n", x_a);
	unsigned int  k_b, x_b;
	k_b = leadingBitPosition(b_abs);
	x_b = b_abs << (n - 1 - k_b);
    //printf("Xb = %x \n", x_b);

    unsigned int tmp, tmp_prim;
    tmp = (1<<(n-1))-1;
    tmp_prim = ((1<<(n-1)) - (1<<(n-w)));

	unsigned int y_a, y_b, tmp_a, tmp_b;
	tmp_a = x_a & tmp;
	y_a = (x_a & tmp_prim);
    y_a = y_a | (1 << (n-w));
    //printf("Ya = %x \n", y_a);

	tmp_b = x_b & tmp;
	y_b = x_b & tmp_prim;
    y_b = y_b | (1 << (n-w));
	//printf("Yb = %x \n", y_b);

	// We truncate mantissa
	unsigned int y_l,inc;
	inc = (1 << (n-w));

	y_l = (y_a + y_b + inc) & tmp;
	// We set the LSB of k_a and k_b to zero


	unsigned int k_l;

	int zero = (a == 0) || (b == 0)   ;
	k_l = k_a + k_b + (((y_a + y_b + inc) & (tmp+1)) >> (n - 1));

	double m;
	unsigned int p_abs;
	m = (double)y_l / (1 << 15);

	p_abs = (unsigned int)((1 + m)*(1 << k_l));
	int p;
	p = (sgn_a ^ sgn_b)? -p_abs-1 : p_abs;
	p = p*(1-zero);
	return p;
}

torch::Tensor mat_mult(torch::Tensor mat1, torch::Tensor mat2, int64_t w) {
    float mult[mat1.size(0)][mat2.size(1)];
    float* a = mat1.data_ptr<float>();
    float* b = mat2.data_ptr<float>();
    int i, j, k;
    // Initializing elements of matrix mult to 0.
    for(i = 0; i < mat1.size(0); i++)
        for(j = 0; j < mat2.size(1); j++)
        {
            mult[i][j] = 0;
        }

    // Multiplying matrix a and b and storing in array mult.
    for(i = 0; i < mat1.size(0); ++i)
        for(j = 0; j < mat2.size(1); ++j)
            for(k = 0; k < mat1.size(1); ++k)
            {
                mult[i][j] += (float) DRALM((int) mat1[i][k].item().to<float>(), (int) mat2[k][j].item().to<float>(), w);
            }

    torch::Tensor output = torch::from_blob(mult, /*sizes=*/{mat1.size(0), mat2.size(1)});
    return output.clone();
  // END output_tensor
}

TORCH_LIBRARY(my_ops, m) {
  m.def("mat_mult", mat_mult);
}