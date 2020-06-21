
main: simple.cu
	nvcc -O3 -g simple.cu -o main
main2: blocked.cu
	nvcc -O3 -g blocked.cu -o main2
main3: shared_memory_blocked.cu
	nvcc -O3 -g shared_memory_blocked.cu -o main3
clean:
	rm -f main
	rm -f main2
	rm -f main3