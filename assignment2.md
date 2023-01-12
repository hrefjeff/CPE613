# Why is the first parameter listed in the cudaMalloc function signature of type void **? Starting here, explain how a call to cudaMalloc works to "return" a pointer to GPU memory.

I'll start with a little bit of setup then get to the answer. In C, Malloc has a return type of void. This is just a way that C passes back an address and lets the programmer decide what to do with it. For example, 

```c
int* x = (int *) malloc(sizeof(int));
// is comparable to
int* x = malloc(sizeof(int));

```

they both work, but the first one says "yea, i know what im doing because i know what is returned from malloc will be type casted".

The return value of cudamalloc is an integer error code instead of a pointer to the memory block like in C. The first parameter in cudaMalloc is designed to allow a programmer to specify where to 
