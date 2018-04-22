

// Celem tego programu jest prezentacja pomiaru i analizy
//efektywnosci programu za pomoc¹  CodeAnalyst(tm).
// Implementacja mno¿enia macierzy jest realizowana za pomoca typowego
// algorytmu podrêcznikowego.
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <time.h>
#include <windows.h>
#include "omp.h"

#define USE_MULTIPLE_THREADS true
#define MAXTHREADS 128
int NumThreads;
double start;

static const int ROWS = 1000;     // liczba wierszy macierzy
static const int COLUMNS = 1000;  // lizba kolumn macierzy

float matrix_a[ROWS][COLUMNS];    // lewy operand
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik

FILE *result_file;

void initialize_matrices()
{
	// zdefiniowanie zawarosci poczatkowej macierzy
	//#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_a[i][j] = (float)rand() / RAND_MAX;
			matrix_b[i][j] = (float)rand() / RAND_MAX;
			matrix_r[i][j] = 0.0;
		}
	}
}

void initialize_matricesZ()
{
	// zdefiniowanie zawarosci poczatkowej macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_r[i][j] = 0.0;
		}
	}
}
void print_result()
{
	// wydruk wyniku
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			fprintf(result_file, "%6.4f ", matrix_r[i][j]);
		}
		fprintf(result_file, "\n");
	}
}

void multiply_matrices_IJK()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			float sum = 0.0;
			for (int k = 0; k < COLUMNS; k++) {
				sum = sum + matrix_a[i][k] * matrix_b[k][j];
			}
			matrix_r[i][j] = sum;
		}
	}
}

void multiply_matrices_IKJ()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int i = 0; i < ROWS; i++)
		for (int k = 0; k < COLUMNS; k++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_seq()
{
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_before_k()
{
#pragma omp parallel 
#pragma omp for
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_before_i()
{
#pragma omp parallel 
	for (int k = 0; k < COLUMNS; k++)
#pragma omp for
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void multiply_matrices_KIJ_before_j()
{
#pragma omp parallel 
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
#pragma omp for
			for (int j = 0; j < COLUMNS; j++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}


void multiply_matrices_KIJ_before_k_atomic()
{
#pragma omp parallel 
#pragma omp for
	for (int k = 0; k < COLUMNS; k++)
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
#pragma omp atomic
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}
void multiply_matrices_KIJ_before_k_reduct()
{
#pragma omp parallel
	{
		float matrix_r_private[ROWS][COLUMNS];    // prywatna kopia
		for (int i = 0; i < ROWS; i++)
			for (int j = 0; j < COLUMNS; j++)
				matrix_r_private[i][j] = 0.0;


#pragma omp for
		for (int k = 0; k < COLUMNS; k++)
			for (int i = 0; i < ROWS; i++)
				for (int j = 0; j < COLUMNS; j++)
					//	matrix_r_private[i][j] += 2.0;
					matrix_r_private[i][j] += matrix_a[i][k] * matrix_b[k][j];


#pragma omp critical
		{
			for (int i = 0; i < ROWS; i++)
				for (int j = 0; j < COLUMNS; j++)
					matrix_r[i][j] += matrix_r_private[i][j];
		}
	}


}

void oli() {
	int aa[] = { 84, 30, 95, 94, 36, 73, 52, 23, 2, 13 };
	int S[10] = { 0 };
#pragma omp parallel
	{
		int S_private[10] = { 0 };
#pragma omp for
		for (int n = 0; n < 10; ++n) {
			for (int m = 0; m <= n; ++m) {
				S_private[n] += aa[m];
			}
		}
#pragma omp critical
		{
			for (int n = 0; n < 10; ++n) {
				S[n] += S_private[n];
			}
		}
	}
}


float suma() {
	float s = 0.0;
	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLUMNS; j++)
			s += matrix_r[i][j];
	return s;
}
void multiply_matrices_JIK()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int j = 0; j < COLUMNS; j++) {
		for (int i = 0; i < ROWS; i++) {
			float sum = 0.0;
			for (int k = 0; k < COLUMNS; k++) {
				sum = sum + matrix_a[i][k] * matrix_b[k][j];
			}
			matrix_r[i][j] = sum;
		}
	}
}
void multiply_matrices_JKI()
{
	// mnozenie macierzy
#pragma omp parallel for
	for (int j = 0; j < COLUMNS; j++)
		for (int k = 0; k < COLUMNS; k++)
			for (int i = 0; i < ROWS; i++)
				matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}


void print_elapsed_time()
{
	double elapsed;
	double resolution;

	// wyznaczenie i zapisanie czasu przetwarzania
	elapsed = (double)clock() / CLK_TCK;
	resolution = 1.0 / CLK_TCK;
	printf("Czas: %8.4f sec\n",
		elapsed - start);

	fprintf(result_file,
		"%8.4f\n",
		elapsed - start);
}

int main(int argc, char* argv[])
{
	for(int i = 0; i < 10; i++){
	//	 start = (double) clock() / CLK_TCK ;
	if ((result_file = fopen("classic.txt", "a")) == NULL) {
		fprintf(stderr, "nie mozna otworzyc pliku wyniku \n");
		perror("classic");
		return(EXIT_FAILURE);
	}


	//Determine the number of threads to use
	if (USE_MULTIPLE_THREADS) {
		SYSTEM_INFO SysInfo;
		GetSystemInfo(&SysInfo);
		NumThreads = SysInfo.dwNumberOfProcessors;
		if (NumThreads > MAXTHREADS)
			NumThreads = MAXTHREADS;
	}
	else
		NumThreads = 1;
	fprintf(result_file, "\n", NumThreads);
	printf("liczba watkow  = %d\n\n", NumThreads);
	printf("KIJ_seq\n");
	printf("KIJ_before_k\n");
	printf("KIJ_before_k_atomic\n");
	printf("KIJ_before_k_reduct\n");
	printf("KIJ_before_i\n");
	printf("KIJ_before_j\n");

	initialize_matrices();

	start = (double)clock() / CLK_TCK;
	multiply_matrices_IKJ();

	print_elapsed_time();
	printf("Suma: %f\n\n", suma());


	fclose(result_file);

}
	return(0);
}
