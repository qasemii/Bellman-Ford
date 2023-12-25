C_SOURCES = $(wildcard src/*.c *.c)
HEADERS = $(wildcard inc/*.h *.h)
OBJ = ${C_SOURCES:.c=.o}
CFLAGS = -fopenmp

MAIN = main
CC = /usr/bin/gcc
LINKER = /usr/bin/ld

run: ${MAIN}
	./${MAIN}

main: ${OBJ}
	${CC} ${CFLAGS} $^ -o $@ -lm

# Generic rules
%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $< -o $@ -lm

clean:
	rm src/*.o *.o${MAIN}