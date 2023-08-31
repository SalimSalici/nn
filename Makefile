CC = "gcc"
CFLAGS += -Wall -O3 -std=gnu11 -mavx2 -I./include
LDFLAGS += -L./lib -lm -lraylib -l:libopenblas.a -lwinmm -lgdi32 -lopengl32

main: mat.c helper.c mnist_loader.c main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: mat.c helper.c mnist_loader.c test.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

cifar: mat.c helper.c mnist_loader.c cifar.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

cifar2: mat.c helper.c mnist_loader.c cifar2.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

autoencoder: mat.c helper.c mnist_loader.c autoencoder.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)