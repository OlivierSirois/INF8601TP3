/*
 * heatsim.c
 *
 *  Created on: 2011-11-17
 *      Author: francis
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

#include "config.h"
#include "part.h"
#include "grid.h"
#include "cart.h"
#include "image.h"
#include "heat.h"
#include "memory.h"
#include "util.h"

#define PROGNAME "heatsim"
#define DEFAULT_OUTPUT_PPM "heatsim.png"
#define DEFAULT_DIMX 1
#define DEFAULT_DIMY 1
#define DEFAULT_ITER 100
#define MAX_TEMP 1000.0
#define DIM_2D 2

int offset_from_coords(int x, int y, int width);

typedef struct ctx {
	cart2d_t *cart;
	grid_t *global_grid;
	grid_t *curr_grid;
	grid_t *next_grid;
	grid_t *heat_grid;
	int numprocs;
	int rank;
	MPI_Comm comm2d;
	FILE *log;
	int verbose;
	int dims[DIM_2D];
	int isperiodic[DIM_2D];
	int coords[DIM_2D];
	int grid_coords[DIM_2D];
	int Xsize;
	int Ysize;
	int reorder;
	int north_peer;
	int south_peer;
	int east_peer;
	int west_peer;
	int grid_width;
	int grid_height;
	MPI_Datatype vector;
} ctx_t;

typedef struct command_opts {
	int dimx;
	int dimy;
	int iter;
	char *input;
	char *output;
	int verbose;
} opts_t;

static opts_t *global_opts = NULL;

__attribute__((noreturn))
static void usage(void) {
	fprintf(stderr, PROGNAME " " VERSION " " PACKAGE_NAME "\n");
	fprintf(stderr, "Usage: " PROGNAME " [OPTIONS] [COMMAND]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  --help	this help\n");
	fprintf(stderr, "  --iter	number of iterations to perform\n");
	fprintf(stderr, "  --dimx	2d decomposition in x dimension\n");
	fprintf(stderr, "  --dimy	2d decomposition in y dimension\n");
	fprintf(stderr, "  --input  png input file\n");
	fprintf(stderr, "  --output ppm output file\n");
	fprintf(stderr, "\n");
	exit(EXIT_FAILURE);
}

static void dump_opts(struct command_opts *opts) {
	printf("%10s %s\n", "option", "value");
	printf("%10s %d\n", "dimx", opts->dimx);
	printf("%10s %d\n", "dimy", opts->dimy);
	printf("%10s %d\n", "iter", opts->iter);
	printf("%10s %s\n", "input", opts->input);
	printf("%10s %s\n", "output", opts->output);
	printf("%10s %d\n", "verbose", opts->verbose);
}

void default_int_value(int *val, int def) {
	if (*val == 0)
		*val = def;
}

static int parse_opts(int argc, char **argv, struct command_opts *opts) {
	int idx;
	int opt;
	int ret = 0;

	struct option options[] = { { "help", 0, 0, 'h' },
			{ "iter", 1, 0, 'r' }, { "dimx", 1, 0, 'x' }, { "dimy",
					1, 0, 'y' }, { "input", 1, 0, 'i' }, {
					"output", 1, 0, 'o' }, { "verbose", 0,
					0, 'v' }, { 0, 0, 0, 0 } };

	memset(opts, 0, sizeof(struct command_opts));

	while ((opt = getopt_long(argc, argv, "hvx:y:l:", options, &idx)) != -1) {
		switch (opt) {
		case 'r':
			opts->iter = atoi(optarg);
			break;
		case 'y':
			opts->dimy = atoi(optarg);
			break;
		case 'x':
			opts->dimx = atoi(optarg);
			break;
		case 'i':
			if (asprintf(&opts->input, "%s", optarg) < 0)
				goto err;
			break;
		case 'o':
			if (asprintf(&opts->output, "%s", optarg) < 0)
				goto err;
			break;
		case 'h':
			usage();
			break;
		case 'v':
			opts->verbose = 1;
			break;
		default:
			printf("unknown option %c\n", opt);
			ret = -1;
			break;
		}
	}

	/* default values*/
	default_int_value(&opts->iter, DEFAULT_ITER);
	default_int_value(&opts->dimx, DEFAULT_DIMX);
	default_int_value(&opts->dimy, DEFAULT_DIMY);
	if (opts->output == NULL)
		if (asprintf(&opts->output, "%s", DEFAULT_OUTPUT_PPM) < 0)
			goto err;
	if (opts->input == NULL) {
		fprintf(stderr, "missing input file");
		goto err;
	}

	if (opts->dimx == 0 || opts->dimy == 0) {
		fprintf(stderr,
				"argument error: dimx and dimy must be greater than 0\n");
		ret = -1;
	}

	if (opts->verbose)
		dump_opts(opts);
	global_opts = opts;
	return ret;
	err:
	FREE(opts->input);
	FREE(opts->output);
	return -1;
}

FILE *open_logfile(int rank) {
	char str[255];
	sprintf(str, "out-%d", rank);
	FILE *f = fopen(str, "w+");
	return f;
}

ctx_t *make_ctx() {
	ctx_t *ctx = (ctx_t *) calloc(1, sizeof(ctx_t));
	return ctx;
}

void free_ctx(ctx_t *ctx) {
	if (ctx == NULL)
		return;
	free_grid(ctx->global_grid);
	free_grid(ctx->curr_grid);
	free_grid(ctx->next_grid);
	free_grid(ctx->heat_grid);
	free_cart2d(ctx->cart);
	if (ctx->log != NULL) {
		fflush(ctx->log);
		fclose(ctx->log);
	}
	FREE(ctx);
}

int init_ctx(ctx_t *ctx, opts_t *opts) {

	ctx->numprocs = opts->dimx * opts->dimy;
	MPI_Comm_size(MPI_COMM_WORLD, &ctx->numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
	//printf("numprocs after %d \n", ctx->numprocs);
	if (opts->dimx * opts->dimy != ctx->numprocs) {
		fprintf(stderr,
				"2D decomposition blocks must equal number of process\n");
		printf("the Dimensions required are = %d \n", ctx->dims[0]*ctx->dims[1]);
		goto err;
	}
	ctx->log = open_logfile(ctx->rank);
	ctx->verbose = opts->verbose;
	ctx->dims[0] = opts->dimx;
	ctx->dims[1] = opts->dimy;
	ctx->isperiodic[0] = 1;
	ctx->isperiodic[1] = 1;
	ctx->reorder = 0;
	grid_t *new_grid = NULL;
	//printf("This is the number of processes: %d\n", ctx->numprocs);

	MPI_Cart_create(MPI_COMM_WORLD,2,ctx->dims,ctx->isperiodic,ctx->reorder,&ctx->comm2d);
	MPI_Cart_shift(ctx->comm2d,1,1,&ctx->north_peer,&ctx->south_peer);
	MPI_Cart_shift(ctx->comm2d,0,1,&ctx->west_peer,&ctx->east_peer);
	MPI_Cart_coords(ctx->comm2d,ctx->rank,2,ctx->coords);
	//printf("we are at : %d, %d \n", ctx->coords[0], ctx->coords[1]);
	/*
	 * FIXME: le processus rank=0 charge l'image
	 * ge du disque
	 * et transfert chaque section aux autres processus
	 */
	//MPI_Request *req = calloc(4*ctx->numprocs, sizeof(MPI_Request));
	MPI_Status *status = calloc(16*ctx->numprocs, sizeof(MPI_Status));
	ctx->Xsize = opts->dimx;
	ctx->Ysize = opts->dimy;
	

	//printf("we are here\n");
	if(ctx->rank == 0){
		 /* load input image */
		image_t *image = load_png(opts->input);
		if (image == NULL)
			goto err;
	 
		/* select the red channel as the heat source */
		ctx->global_grid = grid_from_image(image, CHAN_RED);
	 
		/* grid is normalized to one, multiply by MAX_TEMP */
		grid_multiply(ctx->global_grid, MAX_TEMP);
		/* 2D decomposition */
		ctx->Xsize = ctx->global_grid->width;
		ctx->Ysize = ctx->global_grid->height;
		ctx->cart = make_cart2d(ctx->global_grid->width, ctx->global_grid->height, opts->dimx, opts->dimy);
		cart2d_grid_split(ctx->cart, ctx->global_grid);

		printf("salut");
		/*
		 * FIXME: send grid dimensions and data
		 * Comment traiter le cas de rank=0 ?
		 */		
		
		int coords[DIM_2D];
		MPI_Cart_coords(ctx->comm2d, 0, DIM_2D, coords);
		ctx->grid_coords[0] = ctx->cart->pos[0][coords[0]];
		ctx->grid_coords[1] = ctx->cart->pos[1][coords[1]];

		for(int i = 1; i < ctx->numprocs;i++)
		{
			
			int coords[DIM_2D];
			MPI_Cart_coords(ctx->comm2d, i, DIM_2D, coords);
			grid_t *grid = cart2d_get_grid(ctx->cart, coords[0], coords[1]);	
			
			
			MPI_Send(&grid->width, 1, MPI_INTEGER, i, i * 4 +0, ctx->comm2d);//, &req[(i-1)*4+0]);
			MPI_Send(&grid->height, 1, MPI_INTEGER, i, i * 4  + 1, ctx->comm2d);//, &req[(i-1)*4+1]);
			MPI_Send(&grid->padding, 1 , MPI_INTEGER, i, i * 4  + 2, ctx->comm2d);//, &req[(i-1)*4+2]);
			MPI_Send(grid->dbl, grid->pw*grid->ph, MPI_DOUBLE, i, i * 4  + 3, ctx->comm2d);//, &req[(i-1)*4+3]);
			MPI_Send(&ctx->cart->pos[0][coords[0]], 1, MPI_INTEGER, i, i*4+4, ctx->comm2d);
			MPI_Send(&ctx->cart->pos[1][coords[1]], 1, MPI_INTEGER, i, i*4+5, ctx->comm2d);
			MPI_Send(&ctx->Xsize, 1, MPI_INTEGER, i, i*4+6, ctx->comm2d);
			MPI_Send(&ctx->Ysize, 1, MPI_INTEGER, i, i*4+7, ctx->comm2d);
			

		}
		
		//MPI_Cart_coords(ctx->comm2d, ctx->rank, DIM_2D, coords);
		new_grid = cart2d_get_grid(ctx->cart, coords[0], coords[1]);
	}
	else
	{
	 	/*
		* FIXME: receive dimensions of the grid
		* store into new_grid
		*/
	 	int width, height, padding;
		MPI_Recv(&width, 1, MPI_INTEGER, 0, ctx->rank*4+0, ctx->comm2d, &status[0]);
		MPI_Recv(&height, 1, MPI_INTEGER, 0, ctx->rank*4+1, ctx->comm2d, &status[1]);
		MPI_Recv(&padding, 1, MPI_INTEGER, 0, ctx->rank*4+2, ctx->comm2d, &status[2]);
		new_grid = make_grid(width, height, padding);
		MPI_Recv(new_grid->dbl, new_grid->pw*new_grid->ph, MPI_DOUBLE, 0, ctx->rank * 4 +3, ctx->comm2d, &status[3]);
		MPI_Recv(&ctx->grid_coords[0], 1, MPI_INTEGER, 0, ctx->rank*4+4, ctx->comm2d, &status[4]);
		MPI_Recv(&ctx->grid_coords[1], 1, MPI_INTEGER, 0, ctx->rank*4+5, ctx->comm2d, &status[5]);
		MPI_Recv(&ctx->Xsize, 1, MPI_INTEGER, 0, ctx->rank*4+6, ctx->comm2d, &status[6]);
		//printf("test:%d", ctx->cart->pos[0][coords[3]]);
		MPI_Recv(&ctx->Ysize, 1, MPI_INTEGER, 0, ctx->rank*4+7, ctx->comm2d, &status[7]);
		

	}


	
	/* Utilisation temporaire de global_grid */
	//new_grid = ctx->global_grid;
	//printf("Nous sommes ici\n");
	if (new_grid == NULL)
		goto err;
	/* set padding required for Runge-Kutta */
	
	ctx->curr_grid = grid_padding(new_grid, 1);
	ctx->next_grid = grid_padding(new_grid, 1);
	ctx->heat_grid = grid_padding(new_grid, 1);
	//free_grid(new_grid);

	/* FIXME: create type vector to exchange columns */
	
	MPI_Type_vector(ctx->curr_grid->height,1,ctx->curr_grid->pw,MPI_DOUBLE,&ctx->vector);
	MPI_Type_commit(&ctx->vector);
	MPI_Barrier(MPI_COMM_WORLD);

	return 0;
	err: return -1;
	
}

void dump_ctx(ctx_t *ctx) {
	fprintf(ctx->log, "*** CONTEXT ***\n");
	fprintf(ctx->log, "rank=%d\n", ctx->rank);
	fprintf(ctx->log, "north=%d south=%d west=%d east=%d \n",
			ctx->north_peer, ctx->south_peer,
			ctx->east_peer, ctx->west_peer);
	fprintf(ctx->log, "***************\n");
}

void exchng2d(ctx_t *ctx) {
	/*
	 *  FIXME: Echanger les bordures avec les voisins
	 * 4 echanges doivent etre effectues
	 */

	 
	 grid_t *grid = ctx->next_grid;	 
	 int width = grid->pw;
	 int height = grid->ph;
	 
	 int* data = grid->data;
	 int coords[2] ;
	 int north_coords[2]; 
	 int south_coords[2];
	 int east_coords[2];
	 int west_coords[2];
	 ctx->grid_width = 0;
	 ctx->grid_height = 0;

	 MPI_Datatype ns_transfer;
	 MPI_Datatype ew_transfer;
	 
	 //MPI_Request *req = calloc(4*ctx->numprocs, sizeof(MPI_Request));
	 MPI_Status *status = calloc(4*ctx->numprocs, sizeof(MPI_Status));
	 
	 //we obtain the coords on our global for all of our peers
	 MPI_Cart_coords(ctx->comm2d, ctx->north_peer,2, north_coords);
	 MPI_Cart_coords(ctx->comm2d, ctx->rank, 2, coords);
	 MPI_Cart_coords(ctx->comm2d, ctx->south_peer, 2, south_coords);
	 MPI_Cart_coords(ctx->comm2d, ctx->east_peer, 2, east_coords);
	 MPI_Cart_coords(ctx->comm2d, ctx->west_peer, 2, west_coords);

	 
	 /**/
	 if(width > ctx->Xsize){
		 width = ctx->Xsize;
	 } else if(height > ctx->Ysize){
		 height = ctx->Ysize;
	 }	 
	 //making the cart field for all of our nodes.
	 if(ctx->rank != 0){
		ctx->cart = make_cart2d(ctx->Xsize, ctx->Ysize, ctx->dims[0], ctx->dims[1]);
	 	cart2d_grid_split(ctx->cart, ctx->global_grid);
	 } else{
		 //printf("width = %d, height = %d\n",width, height);
	 }
	 

	 
	 //assigning values to our borders
	 //our addresses to specify the start and end point of the borders on our origin on our grid
	 int addr[3] = {ctx->grid_coords[0], ctx->grid_coords[1], offset_from_coords(ctx->grid_coords[0], ctx->grid_coords[1], ctx->Xsize)} ;
	 //int north_addr[3] = {ctx->cart->pos[0][north_coords[0]], ctx->cart->pos[1][north_coords[1]], ctx->cart->pos[1][north_coords[1]] * ctx->Xsize + ctx->cart->pos[0][north_coords[0]]};
	 int south_addr[3] = {ctx->cart->pos[0][south_coords[0]], ctx->cart->pos[1][south_coords[1]], ctx->cart->pos[1][south_coords[1]] * ctx->Xsize + ctx->cart->pos[0][south_coords[0]]};
	 int east_addr[3] = {ctx->cart->pos[0][east_coords[0]], ctx->cart->pos[1][east_coords[1]], ctx->cart->pos[1][east_coords[1]] * ctx->Xsize + ctx->cart->pos[0][east_coords[0]]};
	 //int west_addr[3] = {ctx->cart->pos[0][west_coords[0]], ctx->cart->pos[1][west_coords[1]], ctx->cart->pos[1][west_coords[1]] * ctx->Xsize + ctx->cart->pos[0][west_coords[0]]};
	 //asd
	 if (ctx->dims[0] ==1){
		ctx->grid_width = ctx->Xsize; 
	 } else if(((east_addr[0] - addr[0]) < 0) && (ctx->dims[0] > 1)){
		ctx->grid_width = ctx->Xsize - addr[0];
	 } else if(ctx->dims[0] > 1){
		ctx->grid_width = east_addr[0] - addr[0];
	 }
	 if(ctx->dims[1] == 1){
		 ctx->grid_height = ctx->Ysize;
	 } else if(((south_addr[1] - addr[1]) < 0) && (ctx->dims[1] > 1)){
		 ctx->grid_height = ctx->Ysize - addr[1];
	 } else if(ctx->dims[1] > 1){
		 ctx->grid_height = south_addr[1] - addr[1];
	 }
	 if((ctx->grid_height == 0) || (ctx->grid_width == 0)){
		 printf("we have an error, cannot compute ctx->grid_height or ctx->grid_width on node %d", ctx->rank);
	 }
	 //printf("my pos : %d, south pos: %d, grid height : %d\n",addr[1], south_addr[1], ctx->grid_height);
	 MPI_Type_contiguous(ctx->grid_width, MPI_INTEGER, &ns_transfer);
	 MPI_Type_vector(ctx->grid_height, 1, ctx->Xsize, MPI_INTEGER, &ew_transfer);
	 MPI_Type_commit(&ns_transfer);
	 MPI_Type_commit(&ew_transfer);

	 int* send_south =(int*)data +  offset_from_coords(addr[0], addr[1] + ctx->grid_height-2, ctx->Xsize);
	 int* send_north = (int*)data + offset_from_coords(addr[0], addr[1] + 1, ctx->Xsize);
	 int* send_east =(int*)data +  offset_from_coords(addr[0] + ctx->grid_width - 2, addr[1], ctx->Xsize);
	 int* send_west = (int*)data + offset_from_coords(addr[0] + 1, addr[1], ctx->Xsize);
	 int* recv_south =(int*)data +  offset_from_coords(addr[0], addr[1] + ctx->grid_height - 1, ctx->Xsize);
	 int* recv_north =(int*)data +  offset_from_coords(addr[0], addr[1], ctx->Xsize);
	 int* recv_east =(int*)data +  offset_from_coords(addr[0] + ctx->grid_height - 1, addr[1], ctx->Xsize);
	 int* recv_west =(int*)data +  offset_from_coords(addr[0], addr[1], ctx->Xsize);

	

	 
	 MPI_Sendrecv(send_south, 1, ns_transfer, ctx->south_peer, 0, recv_north, 1, ns_transfer, ctx->north_peer, 0, ctx->comm2d, status);
	 
	 MPI_Sendrecv(send_north, 1, ns_transfer, ctx->north_peer, 1, recv_south, 1, ns_transfer, ctx->south_peer, 1, ctx->comm2d, status);
	 
	 MPI_Sendrecv(send_east, 1, ew_transfer, ctx->east_peer, 2, recv_west, 1, ew_transfer, ctx->west_peer, 2, ctx->comm2d, status);
	 
	 MPI_Sendrecv(send_west, 1, ew_transfer, ctx->west_peer, 3, recv_east, 1, ew_transfer, ctx->east_peer, 3, ctx->comm2d, status);
	 //printf("sent north, my rank is %d, my grid addr is %d, my send addr is %d and my recv addr is %d\n my grid width is %d, my grid height is %d, X coord is %d, Y coord is %d\n", ctx->rank, addr[2], offset_send_south, offset_recv_north, ctx->grid_width, ctx->grid_height, addr[0], addr[1]);
	 
	 MPI_Barrier(ctx->comm2d);
	 
	 /* test de fonctionnement des echanges de borders
	 if(ctx->rank == 0){
		 printf("we are here\n");
	 }
	 */

	 

	 
}


int gather_result(ctx_t *ctx, opts_t *opts) {
	//TODO("lab3");

	int ret = 0;
	grid_t *local_grid = grid_padding(ctx->next_grid, 0);
	if (local_grid == NULL)
		goto err;

	grid_t *new_grid = NULL;
	MPI_Status *status = calloc(4*ctx->numprocs, sizeof(MPI_Status));
	/*
	 * FIXME: transfer simulation results from all process to rank=0
	 * use grid for this purpose
	 */
	if(ctx->rank == 0){
		int x = opts->dimx;
		int y = opts->dimy;
		int* grid_widths;

		for(int j = 0; j < y; j++){
			for(int i = 0; i < x; i++){

			}
		}
	}
	if(ctx->rank != 0){
		printf("grid width us %d \n", ctx->grid_height);
	}

	/* now we can merge all data blocks, reuse global_grid */
	cart2d_grid_merge(ctx->cart, ctx->global_grid);
	/* temporairement copie de next_grid */
	//grid_copy(ctx->next_grid, ctx->global_grid);

	done: free_grid(local_grid);
	return ret;
	err: ret = -1;
	goto done;
}

int main(int argc, char **argv) {
	ctx_t *ctx = NULL;
	int rep, ret;
	opts_t opts;

	if (parse_opts(argc, argv, &opts) < 0) {
		printf("Error while parsing arguments\n");
		usage();
	}
	if (opts.verbose)
		dump_opts(&opts);

	MPI_Init(&argc, &argv);
	ctx = make_ctx();
	//printf("made CTX\n");
	if (init_ctx(ctx, &opts) < 0)
		goto err;
	if (opts.verbose)
		dump_ctx(ctx);

	if (ctx->verbose) {
		fprintf(ctx->log, "heat grid\n");
		fdump_grid(ctx->heat_grid, ctx->log);
	}
	//printf("before for\n");
	for (rep = 0; rep < opts.iter; rep++) {
		if (ctx->verbose) {
			fprintf(ctx->log, "iter %d\n", rep);
			fprintf(ctx->log, "start\n");
			fdump_grid(ctx->curr_grid, ctx->log);
		}
		//printf("setting grid");
		grid_set_min(ctx->heat_grid, ctx->curr_grid);
		if (ctx->verbose) {
			fprintf(ctx->log, "grid_set_min\n");
			fdump_grid(ctx->curr_grid, ctx->log);
		}
		//printf("Just before call\n");
		exchng2d(ctx);
		if (ctx->verbose) {
			fprintf(ctx->log, "exchng2d\n");
			fdump_grid(ctx->curr_grid, ctx->log);
		}
		
		heat_diffuse(ctx->curr_grid, ctx->next_grid);
		if (ctx->verbose) {
			fprintf(ctx->log, "heat_diffuse\n");
			fdump_grid(ctx->next_grid, ctx->log);
		}
		
		SWAP(ctx->curr_grid, ctx->next_grid);
		
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (gather_result(ctx, &opts) < 0)
		goto err;

	if (ctx->rank == 0) {
		printf("saving...\n");
		if (save_grid_png(ctx->global_grid, opts.output) < 0) {
			printf("saving failed\n");
			goto err;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	ret = EXIT_SUCCESS;
done:
	free_ctx(ctx);
	MPI_Finalize();
	FREE(opts.input);
	FREE(opts.output);
	return ret;
err:
	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	ret = EXIT_FAILURE;
	goto done;
}

int offset_from_coords(int x, int y, int width){
	return x + y*width;
}
