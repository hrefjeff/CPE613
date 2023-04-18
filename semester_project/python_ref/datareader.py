#!/usr/bin/env python3

import numpy as np

def readGPUDataFile(filename):

    print(f'attempting to read file {filename}')

    fid = open(filename, 'r')
    lines = fid.readlines()
    fid.close()

    currentLine = 0

    numberOfDimensions = int(lines[currentLine].split()[0])
    print(f'this data is {numberOfDimensions}-dimensional')
    currentLine = currentLine + 1

    dimensions = []
    for idx in range(numberOfDimensions):
        dimensions.append(int(lines[currentLine].split()[0]))
        currentLine = currentLine + 1
    print(f'the dimensions are {dimensions}')

    totalNumberOfElements = 1
    for s in dimensions:
        totalNumberOfElements = totalNumberOfElements * s
    print(f'there is a total of {totalNumberOfElements} elements')

    datatype = (lines[currentLine].split()[0])
    currentLine = currentLine + 1
    print(f'the data is of type {datatype}')


    if datatype == 'double':
        data = np.empty((totalNumberOfElements,), dtype='f')
        for idx in range(totalNumberOfElements):
            data[idx] = float(lines[currentLine].split()[0])
            print(f'{data[idx]}')
            currentLine = currentLine + 1
    if datatype == 'complex':
        data = np.empty((totalNumberOfElements,), dtype=np.complex_)
        for idx in range(totalNumberOfElements):
            data[idx] = float(lines[currentLine].split()[0]) + 1.j * float(lines[currentLine].split()[1])
            print(f'{data[idx]}')
            currentLine = currentLine + 1
    else:
        raise Exception('this datatype is not yet supported')

    data = np.reshape(data, tuple(dimensions))

    print(f'successfully read data:\n{data}')

    return data

readGPUDataFile('test.txt')
