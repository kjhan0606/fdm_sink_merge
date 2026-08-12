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

/*
 * Extract gas cells and star particles for selected HR5 PSB galaxies.
 *
 * GALFIND.DATA contains native structures written by the compiler used for
 * the original HR5 catalogue production.  The byte sizes below are measured
 * by the archived typebyte executable.  Field offsets follow the archived
 * ramses.h compiled with OUTPUT_PARTICLE_POTENTIAL.
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

_Static_assert(sizeof(Hr5Metadata) == HR5_META_BYTES, "unexpected metadata layout");

typedef struct {
    const char *data_path;
    const char *list_path;
    const char *galaxy_ids_path;
    const char *output_path;
    int force;
    uint8_t *selected;
    uint8_t *seen;
    size_t selected_size;
    uint64_t requested_count;
} Options;

typedef struct {
    uint64_t halo_count;
    uint64_t galaxy_count;
    uint64_t selected_galaxy_count;
    uint64_t gas_count;
    uint64_t star_count;
    uint64_t invalid_count_records;
    uint64_t particle_count_mismatches;
    uint64_t metadata_sample_mismatches;
    uint64_t gas_mass_mismatches;
    uint64_t stellar_mass_mismatches;
} Summary;

static void usage(FILE *stream, const char *program) {
    fprintf(
        stream,
        "Usage: %s --data GALFIND.DATA --list GALCATALOG.LIST "
        "--galaxy-ids IDS.txt --output PARTICLES.csv [--force]\n",
        program
    );
}

static int parse_options(int argc, char **argv, Options *options) {
    int i;
    memset(options, 0, sizeof(*options));
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--force")) {
            options->force = 1;
        } else if (!strcmp(argv[i], "--data") && i + 1 < argc) {
            options->data_path = argv[++i];
        } else if (!strcmp(argv[i], "--list") && i + 1 < argc) {
            options->list_path = argv[++i];
        } else if (!strcmp(argv[i], "--galaxy-ids") && i + 1 < argc) {
            options->galaxy_ids_path = argv[++i];
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            options->output_path = argv[++i];
        } else if (!strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            return -1;
        }
    }
    return (!options->data_path || !options->list_path ||
            !options->galaxy_ids_path || !options->output_path) ? -1 : 0;
}

static int expand_selection(Options *options, size_t required) {
    size_t new_size;
    uint8_t *selected;
    uint8_t *seen;
    if (required <= options->selected_size) {
        return 0;
    }
    new_size = options->selected_size ? options->selected_size : 1024;
    while (new_size < required) {
        if (new_size > SIZE_MAX / 2) {
            new_size = required;
            break;
        }
        new_size *= 2;
    }
    selected = calloc(new_size, 1);
    seen = calloc(new_size, 1);
    if (!selected || !seen) {
        free(selected);
        free(seen);
        return -1;
    }
    if (options->selected_size) {
        memcpy(selected, options->selected, options->selected_size);
        memcpy(seen, options->seen, options->selected_size);
    }
    free(options->selected);
    free(options->seen);
    options->selected = selected;
    options->seen = seen;
    options->selected_size = new_size;
    return 0;
}

static int load_galaxy_ids(Options *options) {
    FILE *stream = fopen(options->galaxy_ids_path, "r");
    int64_t galaxy_id;
    if (!stream) {
        fprintf(stderr, "Could not open %s: %s\n",
                options->galaxy_ids_path, strerror(errno));
        return -1;
    }
    while (fscanf(stream, " %" SCNd64, &galaxy_id) == 1) {
        size_t required;
        if (galaxy_id < 0 || (uint64_t)galaxy_id >= SIZE_MAX - 1) {
            fprintf(stderr, "Invalid galaxy identifier: %" PRId64 "\n", galaxy_id);
            fclose(stream);
            return -1;
        }
        required = (size_t)galaxy_id + 1;
        if (expand_selection(options, required)) {
            fprintf(stderr, "Could not allocate the galaxy selection\n");
            fclose(stream);
            return -1;
        }
        if (!options->selected[galaxy_id]) {
            options->selected[galaxy_id] = 1;
            options->requested_count++;
        }
    }
    if (!feof(stream) || !options->requested_count) {
        fprintf(stderr, "No valid galaxy identifiers in %s\n",
                options->galaxy_ids_path);
        fclose(stream);
        return -1;
    }
    fclose(stream);
    return 0;
}

static int file_size(const char *path, off_t *size) {
    struct stat status;
    if (stat(path, &status) || !S_ISREG(status.st_mode)) {
        fprintf(stderr, "Could not inspect %s: %s\n", path, strerror(errno));
        return -1;
    }
    *size = status.st_size;
    return 0;
}

static int checked_add(off_t *offset, uint64_t count, uint64_t item_size, off_t limit) {
    uint64_t current;
    uint64_t increment;
    if (*offset < 0 || limit < 0 || (count && item_size > UINT64_MAX / count)) {
        return -1;
    }
    current = (uint64_t)*offset;
    increment = count * item_size;
    if (increment > UINT64_MAX - current || current + increment > (uint64_t)limit) {
        return -1;
    }
    *offset = (off_t)(current + increment);
    return 0;
}

static int read_at(FILE *stream, void *buffer, size_t size, off_t offset,
                   const char *path) {
    int descriptor = fileno(stream);
    size_t completed = 0;
    while (completed < size) {
        ssize_t amount = pread(descriptor, (char *)buffer + completed,
                               size - completed, offset + (off_t)completed);
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

static double read_double(const uint8_t *record, size_t offset) {
    double value;
    memcpy(&value, record + offset, sizeof(value));
    return value;
}

static float read_float(const uint8_t *record, size_t offset) {
    float value;
    memcpy(&value, record + offset, sizeof(value));
    return value;
}

static int32_t read_int32(const uint8_t *record, size_t offset) {
    int32_t value;
    memcpy(&value, record + offset, sizeof(value));
    return value;
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

static int write_header(FILE *output) {
    return fprintf(
        output,
        "galaxy_gid,fof_index,psb_index,particle_type,x_cmpc_h,y_cmpc_h,z_cmpc_h,"
        "mass_msun_h,metallicity,formation_time,initial_mass_code,"
        "cell_size_cmpc_h,density_code,temperature_code,level\n"
    ) < 0 ? -1 : 0;
}

static int write_gas(FILE *output, const uint8_t *record, int64_t galaxy_id,
                     uint64_t halo_index, int32_t psb_index, double *mass_sum) {
    double mass = (double)read_float(record, 80);
    *mass_sum += mass;
    return fprintf(
        output,
        "%" PRId64 ",%" PRIu64 ",%" PRId32 ",gas,"
        "%.17g,%.17g,%.17g,%.17g,%.9g,nan,nan,%.17g,%.17g,%.9g,%" PRId32 "\n",
        galaxy_id, halo_index, psb_index,
        read_double(record, 0), read_double(record, 8), read_double(record, 16),
        mass, (double)read_float(record, 60), read_double(record, 24),
        read_double(record, 48), (double)read_float(record, 56), read_int32(record, 76)
    ) < 0 ? -1 : 0;
}

static int write_star(FILE *output, const uint8_t *record, int64_t galaxy_id,
                      uint64_t halo_index, int32_t psb_index, double *mass_sum) {
    double mass = read_double(record, 48);
    *mass_sum += mass;
    return fprintf(
        output,
        "%" PRId64 ",%" PRIu64 ",%" PRId32 ",star,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,nan,nan,nan,%" PRId32 "\n",
        galaxy_id, halo_index, psb_index,
        read_double(record, 0), read_double(record, 8), read_double(record, 16),
        mass, read_double(record, 72), read_double(record, 64),
        read_double(record, 80), read_int32(record, 120)
    ) < 0 ? -1 : 0;
}

static int extract_selected_galaxy(
    FILE *data,
    FILE *output,
    const Options *options,
    const Hr5Metadata *sub,
    off_t particle_offset,
    off_t data_size,
    int64_t galaxy_id,
    uint64_t halo_index,
    int32_t psb_index,
    Summary *summary
) {
    uint8_t *gas = NULL;
    uint8_t *stars = NULL;
    off_t gas_offset = particle_offset;
    off_t star_offset = particle_offset;
    size_t gas_bytes = (size_t)sub->count[1] * HR5_GAS_BYTES;
    size_t star_bytes = (size_t)sub->count[3] * HR5_STAR_BYTES;
    double gas_mass_sum = 0.0;
    double stellar_mass_sum = 0.0;
    int32_t i;
    if (checked_add(&gas_offset, (uint64_t)sub->count[0], HR5_DM_BYTES, data_size) ||
        checked_add(&star_offset, (uint64_t)sub->count[0], HR5_DM_BYTES, data_size) ||
        checked_add(&star_offset, (uint64_t)sub->count[1], HR5_GAS_BYTES, data_size) ||
        checked_add(&star_offset, (uint64_t)sub->count[2], HR5_SINK_BYTES, data_size)) {
        return -1;
    }
    if (gas_bytes) {
        gas = malloc(gas_bytes);
        if (!gas || read_at(data, gas, gas_bytes, gas_offset, options->data_path)) {
            free(gas);
            return -1;
        }
        for (i = 0; i < sub->count[1]; ++i) {
            if (write_gas(output, gas + (size_t)i * HR5_GAS_BYTES,
                          galaxy_id, halo_index, psb_index, &gas_mass_sum)) {
                free(gas);
                return -1;
            }
        }
    }
    free(gas);
    if (star_bytes) {
        stars = malloc(star_bytes);
        if (!stars || read_at(data, stars, star_bytes, star_offset, options->data_path)) {
            free(stars);
            return -1;
        }
        for (i = 0; i < sub->count[3]; ++i) {
            if (write_star(output, stars + (size_t)i * HR5_STAR_BYTES,
                           galaxy_id, halo_index, psb_index, &stellar_mass_sum)) {
                free(stars);
                return -1;
            }
        }
    }
    free(stars);
    if (sub->count[1] && (!isfinite(gas_mass_sum) ||
        fabs(gas_mass_sum - sub->value[2]) > 1.0e-5 * fmax(fabs(sub->value[2]), 1.0))) {
        summary->gas_mass_mismatches++;
    }
    if (sub->count[3] && (!isfinite(stellar_mass_sum) ||
        fabs(stellar_mass_sum - sub->value[4]) > 1.0e-9 * fmax(fabs(sub->value[4]), 1.0))) {
        summary->stellar_mass_mismatches++;
    }
    summary->gas_count += (uint64_t)sub->count[1];
    summary->star_count += (uint64_t)sub->count[3];
    summary->selected_galaxy_count++;
    return 0;
}

static int run_extraction(const Options *options, FILE *output, Summary *summary) {
    FILE *list = NULL;
    FILE *data = NULL;
    off_t list_size = 0;
    off_t data_size = 0;
    off_t data_offset = 0;
    int64_t galaxy_id = 0;
    int result = -1;
    if (file_size(options->list_path, &list_size) ||
        file_size(options->data_path, &data_size) || list_size <= 0 ||
        list_size % HR5_META_BYTES) {
        return -1;
    }
    list = fopen(options->list_path, "rb");
    data = fopen(options->data_path, "rb");
    if (!list || !data || write_header(output)) {
        fprintf(stderr, "Could not open the GALFIND inputs or output header\n");
        goto cleanup;
    }
    while (ftello(list) < list_size) {
        Hr5Metadata halo;
        Hr5Metadata saved_halo;
        int32_t psb_index;
        if (fread(&halo, sizeof(halo), 1, list) != 1 || halo.count[0] < 0) {
            fprintf(stderr, "Invalid FoF metadata at halo %" PRIu64 "\n",
                    summary->halo_count);
            goto cleanup;
        }
        if ((summary->halo_count < 4 || summary->halo_count % 100000 == 0) &&
            (read_at(data, &saved_halo, sizeof(saved_halo), data_offset,
                     options->data_path) || memcmp(&halo, &saved_halo, sizeof(halo)))) {
            summary->metadata_sample_mismatches++;
            goto cleanup;
        }
        if (checked_add(&data_offset, 1, sizeof(halo), data_size)) {
            goto cleanup;
        }
        for (psb_index = 0; psb_index < halo.count[0]; ++psb_index) {
            Hr5Metadata sub;
            int selected;
            if (fread(&sub, sizeof(sub), 1, list) != 1 ||
                validate_subinfo(&sub, summary)) {
                fprintf(stderr, "Invalid PSB metadata at galaxy %" PRId64 "\n", galaxy_id);
                goto cleanup;
            }
            if (checked_add(&data_offset, 1, sizeof(sub), data_size)) {
                goto cleanup;
            }
            selected = (uint64_t)galaxy_id < options->selected_size &&
                       options->selected[galaxy_id];
            if (selected) {
                if (extract_selected_galaxy(data, output, options, &sub, data_offset,
                                            data_size, galaxy_id, summary->halo_count, psb_index,
                                            summary)) {
                    goto cleanup;
                }
                options->seen[galaxy_id] = 1;
            }
            if (checked_add(&data_offset, (uint64_t)sub.count[0], HR5_DM_BYTES, data_size) ||
                checked_add(&data_offset, (uint64_t)sub.count[1], HR5_GAS_BYTES, data_size) ||
                checked_add(&data_offset, (uint64_t)sub.count[2], HR5_SINK_BYTES, data_size) ||
                checked_add(&data_offset, (uint64_t)sub.count[3], HR5_STAR_BYTES, data_size)) {
                goto cleanup;
            }
            summary->galaxy_count++;
            galaxy_id++;
        }
        summary->halo_count++;
        if (summary->halo_count % 100000 == 0) {
            fprintf(stderr, "Read %" PRIu64 " FoF haloes and %" PRIu64
                            " PSB galaxies; selected %" PRIu64 "/%" PRIu64 "\n",
                    summary->halo_count, summary->galaxy_count,
                    summary->selected_galaxy_count, options->requested_count);
        }
    }
    if (data_offset != data_size ||
        summary->selected_galaxy_count != options->requested_count) {
        fprintf(stderr, "GALFIND parse or galaxy-selection count mismatch\n");
        goto cleanup;
    }
    result = 0;

cleanup:
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
    if (load_galaxy_ids(&options)) {
        goto cleanup;
    }
    if (!options.force && access(options.output_path, F_OK) == 0) {
        fprintf(stderr, "Refusing to overwrite %s without --force\n", options.output_path);
        goto cleanup;
    }
    temporary_length = strlen(options.output_path) + 5;
    temporary_path = malloc(temporary_length);
    if (!temporary_path) {
        goto cleanup;
    }
    snprintf(temporary_path, temporary_length, "%s.tmp", options.output_path);
    output = fopen(temporary_path, "wb");
    if (!output || run_extraction(&options, output, &summary) ||
        fflush(output) || fsync(fileno(output)) || fclose(output)) {
        output = NULL;
        goto cleanup;
    }
    output = NULL;
    if (rename(temporary_path, options.output_path)) {
        fprintf(stderr, "Could not rename %s: %s\n", temporary_path, strerror(errno));
        goto cleanup;
    }
    fprintf(
        stderr,
        "Extracted %" PRIu64 " galaxies, %" PRIu64 " gas cells, and %" PRIu64
        " star particles. Gas-mass mismatches: %" PRIu64
        "; stellar-mass mismatches: %" PRIu64 ".\n",
        summary.selected_galaxy_count, summary.gas_count, summary.star_count,
        summary.gas_mass_mismatches, summary.stellar_mass_mismatches
    );
    result = (summary.invalid_count_records || summary.particle_count_mismatches ||
              summary.metadata_sample_mismatches || summary.gas_mass_mismatches ||
              summary.stellar_mass_mismatches) ? EXIT_FAILURE : EXIT_SUCCESS;

cleanup:
    if (output) fclose(output);
    free(temporary_path);
    free(options.selected);
    free(options.seen);
    return result;
}
