#include <stdio.h>

int main() {
    int row = ROW_NUMBER, col = COLUMN_NUMBER;
    int redMatrix[row][col];

    FILE *file = fopen("red.txt", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fscanf(file, "%d", &redMatrix[i][j]);
        }
    }

    fclose(file);

    // redMatrix now contains the contents of the file
    return 0;
}
