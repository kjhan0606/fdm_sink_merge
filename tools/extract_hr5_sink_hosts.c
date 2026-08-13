#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Extract the direct association between HR5 sink particles and PSB galaxies.
 *
 * The legacy GALFIND files contain native C structures written by the Intel
 * compiler used for the HR5 analysis.  In particular, the saved GasType is
 * 128 bytes even though the same header compiled with a current GCC release
 * reports 120 bytes.  The byte sizes below come from the original typebyte
 * executable kept with FoF_PSB_Free_Ver2.
 */

enum {
    HR5_META_BYTES = 112,
    HR5_DM_BYTES = 128,
    HR5_GAS_BYTES = 128,
    HR5_SINK_BYTES = 168,
    HR5_STAR_BYTES = 128,
};

typedef struct {
    int32_t count[6];
    double value[11];
} Hr5Metadata;

typedef struct {
    double value[20];
    int32_t sink_id;
    int32_t padding;
} Hr5Sink;

_Static_assert(sizeof(Hr5Metadata) == HR5_META_BYTES, "unexpected metadata layout");
_Static_assert(sizeof(Hr5Sink) == HR5_SINK_BYTES, "unexpected sink layout");

typedef struct {
    const char *data_path;
    const char *list_path;
    const char *background_path;
    const char *sink_ids_path;
    const char *output_path;
    int output_number;
    double redshift;
    int include_background;
    int force;
    int threads;
    uint8_t *selected_sink_ids;
    size_t selected_sink_id_size;
    uint64_t requested_sink_count;
} Options;

typedef struct {
    uint64_t halo_count;
    uint64_t galaxy_count;
    uint64_t hosted_sink_count;
    uint64_t background_sink_count;
    uint64_t selected_sink_count;
    uint64_t duplicate_sink_count;
    uint64_t invalid_count_records;
    uint64_t particle_count_mismatches;
    uint64_t host_sink_mass_mismatches;
    uint64_t metadata_sample_mismatches;
    int32_t maximum_sink_id;
} Summary;

typedef struct {
    uint8_t *seen;
    size_t size;
} IdSet;

typedef struct {
    FILE *data;
    const char *data_path;
    off_t sink_offset;
    int32_t sink_count;
    Hr5Metadata host;
    uint64_t halo_index;
    int64_t psb_index;
    int64_t galaxy_gid;
    int background;
    Hr5Sink *sinks;
    int read_error;
} SinkRequest;

enum { REQUEST_BATCH_SIZE = 4096 };

typedef struct {
    SinkRequest request[REQUEST_BATCH_SIZE];
    size_t count;
} RequestBatch;

static void usage(FILE *stream, const char *program) {
    fprintf(
        stream,
        "Usage: %s --data GALFIND.DATA --list GALCATALOG.LIST --output TABLE.csv "
        "--output-number N --redshift Z [--background background_ptl] "
        "[--sink-ids IDS.txt] [--threads N] [--force]\n",
        program
    );
}

static int parse_int(const char *text, int *value) {
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_double(const char *text, double *value) {
    char *end = NULL;
    double parsed;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno || end == text || *end != '\0' || !isfinite(parsed)) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int parse_options(int argc, char **argv, Options *options) {
    int i;
    memset(options, 0, sizeof(*options));
    options->output_number = -1;
    options->redshift = NAN;
    options->threads = 8;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--force")) {
            options->force = 1;
        } else if (!strcmp(argv[i], "--data") && i + 1 < argc) {
            options->data_path = argv[++i];
        } else if (!strcmp(argv[i], "--list") && i + 1 < argc) {
            options->list_path = argv[++i];
        } else if (!strcmp(argv[i], "--background") && i + 1 < argc) {
            options->background_path = argv[++i];
            options->include_background = 1;
        } else if (!strcmp(argv[i], "--sink-ids") && i + 1 < argc) {
            options->sink_ids_path = argv[++i];
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            options->output_path = argv[++i];
        } else if (!strcmp(argv[i], "--output-number") && i + 1 < argc) {
            if (parse_int(argv[++i], &options->output_number)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--redshift") && i + 1 < argc) {
            if (parse_double(argv[++i], &options->redshift)) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            if (parse_int(argv[++i], &options->threads) || options->threads < 1 ||
                options->threads > 256) {
                return -1;
            }
        } else if (!strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else {
            return -1;
        }
    }
    if (!options->data_path || !options->list_path || !options->output_path ||
        options->output_number < 0 || !isfinite(options->redshift)) {
        return -1;
    }
    return 0;
}

static int load_sink_ids(Options *options) {
    FILE *stream;
    int32_t sink_id;
    if (!options->sink_ids_path) {
        return 0;
    }
    stream = fopen(options->sink_ids_path, "r");
    if (!stream) {
        fprintf(stderr, "Could not open %s: %s\n", options->sink_ids_path, strerror(errno));
        return -1;
    }
    while (fscanf(stream, " %" SCNd32, &sink_id) == 1) {
        size_t required;
        uint8_t *expanded;
        if (sink_id <= 0) {
            fprintf(stderr, "Invalid sink identifier in %s: %" PRId32 "\n",
                    options->sink_ids_path, sink_id);
            fclose(stream);
            return -1;
        }
        required = (size_t)sink_id + 1;
        if (required > options->selected_sink_id_size) {
            size_t new_size = options->selected_sink_id_size ?
                              options->selected_sink_id_size : 1024;
            while (new_size < required) {
                if (new_size > SIZE_MAX / 2) {
                    new_size = required;
                    break;
                }
                new_size *= 2;
            }
            expanded = realloc(options->selected_sink_ids, new_size);
            if (!expanded) {
                fprintf(stderr, "Could not allocate sink selection array\n");
                fclose(stream);
                return -1;
            }
            memset(expanded + options->selected_sink_id_size, 0,
                   new_size - options->selected_sink_id_size);
            options->selected_sink_ids = expanded;
            options->selected_sink_id_size = new_size;
        }
        if (!options->selected_sink_ids[sink_id]) {
            options->selected_sink_ids[sink_id] = 1;
            options->requested_sink_count++;
        }
    }
    if (!feof(stream)) {
        fprintf(stderr, "Invalid text in sink selection file %s\n", options->sink_ids_path);
        fclose(stream);
        return -1;
    }
    fclose(stream);
    if (!options->requested_sink_count) {
        fprintf(stderr, "No sink identifiers in %s\n", options->sink_ids_path);
        return -1;
    }
    return 0;
}

static int sink_is_selected(const Options *options, int32_t sink_id) {
    if (!options->sink_ids_path) {
        return 1;
    }
    return sink_id > 0 && (size_t)sink_id < options->selected_sink_id_size &&
           options->selected_sink_ids[sink_id];
}

static int file_size(const char *path, off_t *size) {
    struct stat status;
    if (stat(path, &status)) {
        fprintf(stderr, "Could not stat %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (!S_ISREG(status.st_mode)) {
        fprintf(stderr, "%s is not a regular file\n", path);
        return -1;
    }
    *size = status.st_size;
    return 0;
}

static int checked_add(off_t *offset, uint64_t count, uint64_t item_size, off_t limit) {
    uint64_t current;
    uint64_t increment;
    if (*offset < 0 || limit < 0) {
        return -1;
    }
    current = (uint64_t)*offset;
    if (count && item_size > UINT64_MAX / count) {
        return -1;
    }
    increment = count * item_size;
    if (increment > UINT64_MAX - current || current + increment > (uint64_t)limit) {
        return -1;
    }
    *offset = (off_t)(current + increment);
    return 0;
}

static int validate_subinfo(const Hr5Metadata *sub, Summary *summary) {
    int i;
    int64_t sum = 0;
    for (i = 0; i < 5; ++i) {
        if (sub->count[i] < 0) {
            summary->invalid_count_records++;
            return -1;
        }
    }
    sum = (int64_t)sub->count[0] + sub->count[1] + sub->count[2] + sub->count[3];
    if (sum != sub->count[4]) {
        summary->particle_count_mismatches++;
    }
    return 0;
}

static int idset_add(IdSet *set, int32_t sink_id, Summary *summary) {
    size_t required;
    uint8_t *expanded;
    if (sink_id <= 0) {
        return 0;
    }
    required = (size_t)sink_id + 1;
    if (required > set->size) {
        size_t new_size = set->size ? set->size : 1024;
        while (new_size < required) {
            if (new_size > SIZE_MAX / 2) {
                new_size = required;
                break;
            }
            new_size *= 2;
        }
        expanded = realloc(set->seen, new_size);
        if (!expanded) {
            fprintf(stderr, "Could not allocate sink-identifier validation array\n");
            return -1;
        }
        memset(expanded + set->size, 0, new_size - set->size);
        set->seen = expanded;
        set->size = new_size;
    }
    if (set->seen[sink_id]) {
        summary->duplicate_sink_count++;
    }
    set->seen[sink_id] = 1;
    if (sink_id > summary->maximum_sink_id) {
        summary->maximum_sink_id = sink_id;
    }
    return 0;
}

static int read_at(FILE *stream, void *buffer, size_t size, off_t offset, const char *path) {
    int descriptor = fileno(stream);
    size_t completed = 0;
    while (completed < size) {
        ssize_t amount = pread(
            descriptor,
            (char *)buffer + completed,
            size - completed,
            offset + (off_t)completed
        );
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        if (amount <= 0) {
            fprintf(stderr, "Could not read %zu bytes at offset %jd from %s\n",
                    size, (intmax_t)offset, path);
            return -1;
        }
        completed += (size_t)amount;
    }
    return 0;
}

static int write_header(FILE *output) {
    return fprintf(
        output,
        "output,redshift,sink_id,fof_index,psb_index,galaxy_gid,background,"
        "sink_x_cmpc_h,sink_y_cmpc_h,sink_z_cmpc_h,"
        "sink_vx_km_s,sink_vy_km_s,sink_vz_km_s,sink_mass_msun_h,"
        "host_total_mass_msun_h,host_dm_mass_msun_h,host_gas_mass_msun_h,"
        "host_sink_mass_msun_h,host_stellar_mass_msun_h,"
        "host_x_cmpc_h,host_y_cmpc_h,host_z_cmpc_h,"
        "host_vx_km_s,host_vy_km_s,host_vz_km_s,host_sink_count,"
        "host_dm_count,host_gas_count,host_stellar_count,host_particle_count\n"
    ) < 0 ? -1 : 0;
}

static int write_sink(
    FILE *output,
    const Options *options,
    const Hr5Sink *sink,
    const Hr5Metadata *host,
    uint64_t halo_index,
    int64_t psb_index,
    int64_t galaxy_gid,
    int background
) {
    return fprintf(
        output,
        "%d,%.9g,%" PRId32 ",%" PRIu64 ",%" PRId64 ",%" PRId64 ",%d,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32 "\n",
        options->output_number,
        options->redshift,
        sink->sink_id,
        halo_index,
        psb_index,
        galaxy_gid,
        background,
        sink->value[0], sink->value[1], sink->value[2],
        sink->value[3], sink->value[4], sink->value[5], sink->value[6],
        host->value[0], host->value[1], host->value[2], host->value[3], host->value[4],
        host->value[5], host->value[6], host->value[7],
        host->value[8], host->value[9], host->value[10],
        host->count[2], host->count[0], host->count[1], host->count[3], host->count[4]
    ) < 0 ? -1 : 0;
}

static int emit_sink_block(
    const Hr5Sink *sinks,
    int32_t sink_count,
    FILE *output,
    const Options *options,
    const Hr5Metadata *host,
    uint64_t halo_index,
    int64_t psb_index,
    int64_t galaxy_gid,
    int background,
    Summary *summary,
    IdSet *ids
) {
    int32_t i;
    double mass_sum = 0.0;
    for (i = 0; i < sink_count; ++i) {
        mass_sum += sinks[i].value[6];
        if (idset_add(ids, sinks[i].sink_id, summary)) {
            return -1;
        }
        if (sink_is_selected(options, sinks[i].sink_id)) {
            if (write_sink(output, options, &sinks[i], host, halo_index,
                           psb_index, galaxy_gid, background)) {
                return -1;
            }
            summary->selected_sink_count++;
        }
    }
    if (sink_count > 0) {
        double expected = host->value[3];
        double scale = fmax(fabs(expected), 1.0);
        if (!isfinite(mass_sum) || !isfinite(expected) || fabs(mass_sum - expected) > 1.0e-9 * scale) {
            summary->host_sink_mass_mismatches++;
        }
    }
    return 0;
}

static int process_request_batch(
    RequestBatch *batch,
    FILE *output,
    const Options *options,
    Summary *summary,
    IdSet *ids
) {
    size_t request_index;
    int error_count = 0;
#ifdef _OPENMP
    omp_set_num_threads(options->threads);
#pragma omp parallel for schedule(dynamic, 32) reduction(+:error_count)
#endif
    for (request_index = 0; request_index < batch->count; ++request_index) {
        SinkRequest *request = &batch->request[request_index];
        size_t byte_count = (size_t)request->sink_count * sizeof(Hr5Sink);
        request->sinks = malloc(byte_count);
        if (!request->sinks ||
            read_at(request->data, request->sinks, byte_count,
                    request->sink_offset, request->data_path)) {
            request->read_error = 1;
            error_count++;
        }
    }
    if (error_count) {
        for (request_index = 0; request_index < batch->count; ++request_index) {
            free(batch->request[request_index].sinks);
            batch->request[request_index].sinks = NULL;
        }
        return -1;
    }
    for (request_index = 0; request_index < batch->count; ++request_index) {
        SinkRequest *request = &batch->request[request_index];
        if (emit_sink_block(request->sinks, request->sink_count, output, options,
                            &request->host, request->halo_index, request->psb_index,
                            request->galaxy_gid, request->background, summary, ids)) {
            free(request->sinks);
            request->sinks = NULL;
            return -1;
        }
        free(request->sinks);
        request->sinks = NULL;
    }
    batch->count = 0;
    return 0;
}

static int queue_sink_request(
    RequestBatch *batch,
    FILE *data,
    const char *data_path,
    off_t sink_offset,
    int32_t sink_count,
    FILE *output,
    const Options *options,
    const Hr5Metadata *host,
    uint64_t halo_index,
    int64_t psb_index,
    int64_t galaxy_gid,
    int background,
    Summary *summary,
    IdSet *ids
) {
    SinkRequest *request;
    if (batch->count == REQUEST_BATCH_SIZE &&
        process_request_batch(batch, output, options, summary, ids)) {
        return -1;
    }
    request = &batch->request[batch->count++];
    memset(request, 0, sizeof(*request));
    request->data = data;
    request->data_path = data_path;
    request->sink_offset = sink_offset;
    request->sink_count = sink_count;
    request->host = *host;
    request->halo_index = halo_index;
    request->psb_index = psb_index;
    request->galaxy_gid = galaxy_gid;
    request->background = background;
    return 0;
}

static int extract_background(
    FILE *background,
    const Options *options,
    FILE *output,
    uint64_t halo_count,
    Summary *summary,
    IdSet *ids,
    RequestBatch *batch
) {
    off_t offset = 0;
    off_t size = 0;
    uint64_t halo_index;
    if (file_size(options->background_path, &size)) {
        return -1;
    }
    for (halo_index = 0; halo_index < halo_count; ++halo_index) {
        Hr5Metadata sub;
        off_t sink_offset;
        if (read_at(background, &sub, sizeof(sub), offset, options->background_path)) {
            return -1;
        }
        if (checked_add(&offset, 1, sizeof(sub), size) || validate_subinfo(&sub, summary)) {
            fprintf(stderr, "Invalid background record for FoF halo %" PRIu64 "\n", halo_index);
            return -1;
        }
        sink_offset = offset;
        if (checked_add(&sink_offset, (uint64_t)sub.count[0], HR5_DM_BYTES, size) ||
            checked_add(&sink_offset, (uint64_t)sub.count[1], HR5_GAS_BYTES, size)) {
            return -1;
        }
        if (sub.count[2] > 0) {
            if (queue_sink_request(batch, background, options->background_path, sink_offset,
                                   sub.count[2], output, options, &sub, halo_index,
                                   -1, -1, 1, summary, ids)) {
                return -1;
            }
            summary->background_sink_count += (uint64_t)sub.count[2];
        }
        if (checked_add(&offset, (uint64_t)sub.count[0], HR5_DM_BYTES, size) ||
            checked_add(&offset, (uint64_t)sub.count[1], HR5_GAS_BYTES, size) ||
            checked_add(&offset, (uint64_t)sub.count[2], HR5_SINK_BYTES, size) ||
            checked_add(&offset, (uint64_t)sub.count[3], HR5_STAR_BYTES, size)) {
            fprintf(stderr, "Background offset exceeds file size at FoF halo %" PRIu64 "\n",
                    halo_index);
            return -1;
        }
    }
    if (offset != size) {
        fprintf(stderr, "Background file has %jd unparsed bytes\n", (intmax_t)(size - offset));
        return -1;
    }
    if (batch->count && process_request_batch(batch, output, options, summary, ids)) {
        return -1;
    }
    return 0;
}

static int run_extraction(const Options *options, FILE *output, Summary *summary) {
    FILE *list = NULL;
    FILE *data = NULL;
    FILE *background = NULL;
    off_t list_size = 0;
    off_t data_size = 0;
    off_t data_offset = 0;
    int64_t galaxy_gid = 0;
    IdSet ids = {0};
    RequestBatch *batch = NULL;
    int result = -1;

    if (file_size(options->list_path, &list_size) || file_size(options->data_path, &data_size)) {
        return -1;
    }
    if (list_size <= 0 || list_size % HR5_META_BYTES) {
        fprintf(stderr, "Unexpected GALCATALOG.LIST size: %jd bytes\n", (intmax_t)list_size);
        return -1;
    }
    list = fopen(options->list_path, "rb");
    data = fopen(options->data_path, "rb");
    if (!list || !data) {
        fprintf(stderr, "Could not open GALFIND inputs: %s\n", strerror(errno));
        goto cleanup;
    }
    if (options->include_background) {
        background = fopen(options->background_path, "rb");
        if (!background) {
            fprintf(stderr, "Could not open %s: %s\n", options->background_path, strerror(errno));
            goto cleanup;
        }
    }
    batch = calloc(1, sizeof(*batch));
    if (!batch) {
        fprintf(stderr, "Could not allocate the sink-read request batch\n");
        goto cleanup;
    }
    if (write_header(output)) {
        goto cleanup;
    }

    while (ftello(list) < list_size) {
        Hr5Metadata halo;
        Hr5Metadata saved_halo;
        int32_t psb_index;
        if (fread(&halo, sizeof(halo), 1, list) != 1) {
            fprintf(stderr, "Could not read FoF halo metadata at record %" PRIu64 "\n",
                    summary->halo_count);
            goto cleanup;
        }
        if (halo.count[0] < 0) {
            fprintf(stderr, "Negative PSB count at FoF halo %" PRIu64 "\n", summary->halo_count);
            summary->invalid_count_records++;
            goto cleanup;
        }
        if ((summary->halo_count < 4 || summary->halo_count % 100000 == 0) &&
            (read_at(data, &saved_halo, sizeof(saved_halo), data_offset, options->data_path) ||
             memcmp(&halo, &saved_halo, sizeof(halo)))) {
            fprintf(stderr, "GALCATALOG.LIST and GALFIND.DATA differ at FoF halo %" PRIu64
                            "\n", summary->halo_count);
            summary->metadata_sample_mismatches++;
            goto cleanup;
        }
        if (checked_add(&data_offset, 1, sizeof(halo), data_size)) {
            fprintf(stderr, "GALFIND offset exceeds file size at FoF halo %" PRIu64 "\n",
                    summary->halo_count);
            goto cleanup;
        }
        for (psb_index = 0; psb_index < halo.count[0]; ++psb_index) {
            Hr5Metadata sub;
            off_t sink_offset;
            if (fread(&sub, sizeof(sub), 1, list) != 1) {
                fprintf(stderr, "Truncated PSB metadata at FoF halo %" PRIu64 "\n",
                        summary->halo_count);
                goto cleanup;
            }
            if (validate_subinfo(&sub, summary)) {
                fprintf(stderr, "Invalid particle counts at FoF halo %" PRIu64
                                ", PSB %" PRId32 "\n", summary->halo_count, psb_index);
                goto cleanup;
            }
            if (checked_add(&data_offset, 1, sizeof(sub), data_size)) {
                goto cleanup;
            }
            sink_offset = data_offset;
            if (checked_add(&sink_offset, (uint64_t)sub.count[0], HR5_DM_BYTES, data_size) ||
                checked_add(&sink_offset, (uint64_t)sub.count[1], HR5_GAS_BYTES, data_size)) {
                goto cleanup;
            }
            if (sub.count[2] > 0) {
                if (queue_sink_request(batch, data, options->data_path, sink_offset,
                                       sub.count[2], output, options, &sub,
                                       summary->halo_count, psb_index, galaxy_gid, 0,
                                       summary, &ids)) {
                    goto cleanup;
                }
                summary->hosted_sink_count += (uint64_t)sub.count[2];
            }
            if (checked_add(&data_offset, (uint64_t)sub.count[0], HR5_DM_BYTES, data_size) ||
                checked_add(&data_offset, (uint64_t)sub.count[1], HR5_GAS_BYTES, data_size) ||
                checked_add(&data_offset, (uint64_t)sub.count[2], HR5_SINK_BYTES, data_size) ||
                checked_add(&data_offset, (uint64_t)sub.count[3], HR5_STAR_BYTES, data_size)) {
                fprintf(stderr, "GALFIND offset exceeds file size at FoF halo %" PRIu64
                                ", PSB %" PRId32 "\n", summary->halo_count, psb_index);
                goto cleanup;
            }
            summary->galaxy_count++;
            galaxy_gid++;
        }
        summary->halo_count++;
        if (summary->halo_count % 100000 == 0) {
            fprintf(stderr, "Read %" PRIu64 " FoF haloes, %" PRIu64
                            " PSB galaxies, %" PRIu64 " hosted sinks\n",
                    summary->halo_count, summary->galaxy_count, summary->hosted_sink_count);
        }
    }
    if (data_offset != data_size) {
        fprintf(stderr, "GALFIND.DATA has %jd unparsed bytes\n", (intmax_t)(data_size - data_offset));
        goto cleanup;
    }
    if (batch->count && process_request_batch(batch, output, options, summary, &ids)) {
        goto cleanup;
    }
    if (options->include_background &&
        extract_background(background, options, output, summary->halo_count, summary, &ids,
                           batch)) {
        goto cleanup;
    }
    result = 0;

cleanup:
    free(batch);
    free(ids.seen);
    if (background) fclose(background);
    if (data) fclose(data);
    if (list) fclose(list);
    return result;
}

int main(int argc, char **argv) {
    Options options;
    Summary summary = {0};
    FILE *output = NULL;
    char *temporary_path = NULL;
    size_t temporary_length;
    int result = EXIT_FAILURE;

    if (parse_options(argc, argv, &options)) {
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }
    if (load_sink_ids(&options)) {
        free(options.selected_sink_ids);
        return EXIT_FAILURE;
    }
    if (!options.force && access(options.output_path, F_OK) == 0) {
        fprintf(stderr, "Refusing to overwrite %s without --force\n", options.output_path);
        return EXIT_FAILURE;
    }
    temporary_length = strlen(options.output_path) + 5;
    temporary_path = malloc(temporary_length);
    if (!temporary_path) {
        return EXIT_FAILURE;
    }
    snprintf(temporary_path, temporary_length, "%s.tmp", options.output_path);
    output = fopen(temporary_path, "wb");
    if (!output) {
        fprintf(stderr, "Could not create %s: %s\n", temporary_path, strerror(errno));
        goto cleanup;
    }
    if (run_extraction(&options, output, &summary)) {
        goto cleanup;
    }
    if (fflush(output) || fsync(fileno(output)) || fclose(output)) {
        output = NULL;
        fprintf(stderr, "Could not finalize %s\n", temporary_path);
        goto cleanup;
    }
    output = NULL;
    if (rename(temporary_path, options.output_path)) {
        fprintf(stderr, "Could not rename %s to %s: %s\n",
                temporary_path, options.output_path, strerror(errno));
        goto cleanup;
    }
    fprintf(
        stdout,
        "{\"output\":%d,\"redshift\":%.9g,\"fof_halo_count\":%" PRIu64
        ",\"psb_galaxy_count\":%" PRIu64 ",\"hosted_sink_count\":%" PRIu64
        ",\"background_sink_count\":%" PRIu64
        ",\"requested_sink_count\":%" PRIu64 ",\"selected_sink_count\":%" PRIu64
        ",\"duplicate_sink_count\":%" PRIu64
        ",\"particle_count_mismatches\":%" PRIu64
        ",\"host_sink_mass_mismatches\":%" PRIu64
        ",\"metadata_sample_mismatches\":%" PRIu64
        ",\"maximum_sink_id\":%" PRId32 "}\n",
        options.output_number, options.redshift, summary.halo_count, summary.galaxy_count,
        summary.hosted_sink_count, summary.background_sink_count,
        options.requested_sink_count, summary.selected_sink_count,
        summary.duplicate_sink_count, summary.particle_count_mismatches,
        summary.host_sink_mass_mismatches, summary.metadata_sample_mismatches,
        summary.maximum_sink_id
    );
    result = EXIT_SUCCESS;

cleanup:
    if (output) fclose(output);
    if (result != EXIT_SUCCESS && temporary_path) unlink(temporary_path);
    free(options.selected_sink_ids);
    free(temporary_path);
    return result;
}
