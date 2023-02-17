LOCALSAMPLESPATH=/usr/local/cuda-12.0/cuda-samples/Common
ASCSAMPLESPATH=/opt/asn/apps/cuda_11.7.0/samples/common/inc

main: deviceQuery.cpp
	nvcc -I$(LOCALSAMPLESPATH) -o dq.o deviceQuery.cpp

clean:
	rm -f dq.o
