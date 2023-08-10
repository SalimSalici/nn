CC = "gcc"
CFLAGS += -Wall -O3 -std=gnu11 -I./include
LDFLAGS += -L./lib -lm -l:libopenblas.a

main: mat.c helper.c mnist_loader.c main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: mat.c helper.c mnist_loader.c test.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)