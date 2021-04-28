# 02-27-2021
import gzip
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle
import pysam
from models import batch_run_DM_model
from dask import delayed, compute


def reverse_complement(seq):
    base_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    out_seq = ''
    for base in seq[::-1]:
        out_seq += base_dict[base]
    return out_seq


def _get_sequence(coords, ref_seqs):
    sequence = ''
    for (_chr, strand, start, end) in coords:
        ref_seq = ref_seqs[_chr]
        seq = ''
        if start < 0:
            seq += 'N' * (0 - start)
            start = 0
        length = len(ref_seq)
        if end > length:
            seq += ref_seq[start:length].upper()
            seq += 'N' * (end - length)
        else:
            seq += ref_seq[start:end].upper()
        sequence += seq
    return sequence


def get_sequences(intron_coords, ref_seqs, length=100):
    intron_seq_dict = {}
    for (_chr, strand, start, end) in intron_coords:
        coords = []
        # splice site in the center, zero based (-1)
        coords.append((_chr, strand, start - length, start + length))
        coords.append((_chr, strand, end - (length + 1), end + length - 1))
        sequence = _get_sequence(coords, ref_seqs)
        if strand == '-':
            sequence = reverse_complement(sequence)

        intron_seq_dict[(_chr, strand, start, end)] = sequence
    return intron_seq_dict


def get_reference_sequence(file, chr_names):
    fastq_obj = pysam.FastaFile(file)
    ref_seqs = {}
    for _chr in chr_names:
        ref_seqs[_chr] = fastq_obj.fetch(_chr)
    return ref_seqs


def sequence_to_matrix(sequence):
    # R: purine, = A or G
    # Y: pyrimidine, = C or T
    # M is A or C
    base_dict = {'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.], 'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
                'N': [0.25, 0.25, 0.25, 0.25], 'R': [0.5, 0., 0.5, 0.], 'Y': [0., 0.5, 0., 0.5], 'M': [0.5, 0.5, 0, 0]}
    matrix = [base_dict[base] for base in sequence]
    return np.array(matrix).T


def get_seq_matrices(intron_seq_dict):
    intron_matrix_dict = {}
    for intron_coord, seq in intron_seq_dict.items():
        intron_matrix_dict[intron_coord] = sequence_to_matrix(seq)
    return intron_matrix_dict


def generate_sequence_matrices(intron_coords, ref_seqs):
    intron_seq_dict = get_sequences(intron_coords, ref_seqs)
    intron_matrix_dict = get_seq_matrices(intron_seq_dict)
    return intron_matrix_dict


def generate_neural_network_inputs(splice_site_introns_dict, intron_counts_dict, ref_seqs):
    input_dict = defaultdict(list)
    for site, introns in splice_site_introns_dict.items():
        intron_matrix_dict = generate_sequence_matrices(introns, ref_seqs)
        matrices, counts = [], []
        for intron, matrix in intron_matrix_dict.items():
            matrices.append(matrix)
            counts.append(intron_counts_dict[intron])
        size = len(introns)
        input_dict[size].append((np.array(matrices), np.array(counts)))
    return input_dict


def get_info_from_annotation(gtf_file):
    splice_sites = set()
    splice_site_genes_dict = defaultdict(set)
    type_tags = {'nonsense_mediated_decay', 'protein_coding', 'Retained_intron', 'non_coding'}
    chr_names = ['chr{}'.format(i) for i in range(1, 23)] + ['chrX', 'chrY']
    with gzip.open(gtf_file, 'rb') as f:
        for line in f:
            line = line.decode('UTF-8')
            if line != '\n' and not line.startswith('#'):
                items = line.strip().split('\t')
                if items[2] == 'exon':
                    _chr, strand = items[0], items[6]
                    if _chr not in chr_names:
                        continue
                    start, end = int(items[3]), int(items[4])
                    tags = items[8]
                    tag_dict = dict(kv.strip().split(" ") for kv in tags.strip(";").split("; "))
                    if tag_dict['gene_type'][1:-1] == 'protein_coding' and tag_dict['transcript_type'][1:-1] in type_tags:
                        # and tag_dict['transcript_type'][1:-1] == 'protein_coding' :
                        ## only select high quality exons and transcripts
                        # if 'transcript_support_level' not in tag_dict or tag_dict['transcript_support_level'][1:-1] not in '123':
                        #     continue
                        site1, site2 = (_chr, strand, end, 'i'), (_chr, strand, start, 'o')
                        splice_sites.add(site1)
                        splice_sites.add(site2)
                        tran_id, gene_name = tag_dict['transcript_id'][1:-1], tag_dict['gene_name'][1:-1]
                        splice_site_genes_dict[site1].add(gene_name)
                        splice_site_genes_dict[site2].add(gene_name)

    return splice_sites, splice_site_genes_dict


def process_splice_files(splice_list_file, splice_sites):
    splice_files = []
    with open(splice_list_file, 'r') as f:
        for line in f:
            splice_files.append(line.split('\t')[0])

    intron_counts_dict = {}
    selected_introns = set()
    for i, file in enumerate(splice_files):
        with gzip.open(file, 'rb') as f:
            for line in f:
                items = line.decode('UTF-8').strip().split(' ')
                _chr, start, end, count, strand, unique_count = items[:6]
                count, unique_count = int(count), int(unique_count)
                start, end = int(start), int(end)
                if end - start > 10000:
                    continue
                intron = (_chr, strand, start, end)
                if intron not in intron_counts_dict:
                    intron_counts_dict[intron] = [0] * len(splice_files)
                intron_counts_dict[intron][i] = count
                if count >= 3:
                    selected_introns.add(intron)

    # all_introns = list(intron_counts_dict.keys())
    # for intron in all_introns:
    #     if intron not in selected_introns:
    #         del intron_counts_dict[intron]

    splice_site_introns_dict = defaultdict(set)
    for intron in selected_introns:
        _chr, strand, start, end = intron
        site1, site2 = (_chr, strand, start, 'i'), (_chr, strand, end, 'o')
        if site1 in splice_sites:
            splice_site_introns_dict[site1].add(intron)
        if site2 in splice_sites:
            splice_site_introns_dict[site2].add(intron)

    sites = list(splice_site_introns_dict.keys())
    for site in sites:
        if len(splice_site_introns_dict[site]) == 1:
            del splice_site_introns_dict[site]

    return splice_site_introns_dict, intron_counts_dict


def train_test_split(splice_site_introns_dict, splice_site_genes_dict):
    paralogs = set()
    for site, genes in splice_site_genes_dict.items():
        if len(genes) > 1:
            paralogs.update(genes)

    selected_genes = set()
    for site, genes in splice_site_genes_dict.items():
        genes = list(genes)
        if site in splice_site_introns_dict and len(genes) == 1 and genes[0] not in paralogs:
            selected_genes.add(genes[0])

    size = int(len(selected_genes) * 0.1)
    lst = list(selected_genes)
    test_genes = set(random.sample(lst, size))

    train_splice_site_introns_dict = {}
    test_splice_site_introns_dict = {}
    for site, introns in splice_site_introns_dict.items():
        genes = list(splice_site_genes_dict[site])
        if genes[0] in test_genes:
            test_splice_site_introns_dict[site] = introns
        else:
            train_splice_site_introns_dict[site] = introns
    return train_splice_site_introns_dict, test_splice_site_introns_dict, test_genes


def mntjulip_DM_model_processing(model_dir, splice_site_introns_dict, intron_counts_dict, conditions):
    delayed_results = []
    ys, coords, groups = [], [], []
    coords_batches, groups_batches = [], []

    i, batch_size = 0, 2000
    for site, introns in splice_site_introns_dict.items():
        intron_counts = []
        for intron in introns:
            intron_counts.append(intron_counts_dict[intron])
        ys.append(np.array(intron_counts).T)
        groups.append(site)
        coords.append(introns)
        if i > 0 and i % batch_size == 0:
            delayed_results.append(delayed(batch_run_DM_model)(ys, conditions, model_dir, False))
            groups_batches.append(groups)
            coords_batches.append(coords)
            ys, coords, groups = [], [], []
        i += 1

    if len(ys) > 0:
        delayed_results.append(delayed(batch_run_DM_model)(ys, conditions, model_dir, False))
        groups_batches.append(groups)
        coords_batches.append(coords)

    num_workers = 5
    results_batches = list(compute(*delayed_results, traverse=False, num_workers=num_workers, scheduler="processes"))
    return groups_batches, results_batches, coords_batches


def get_conditions():
    conditions = np.zeros([120, 12])
    pre = 0
    for i, j in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 24, 36, 40, 52, 64, 75, 87, 95, 102, 114, 120]):
        conditions[pre:j, i] = 1
        pre = j
    return conditions.astype('uint8')


gtf_file = '/home-2/gyang22@jhu.edu/work/Guangyu/gencode.v37.annotation.gtf.gz'
splice_list_file = Path('/scratch/groups/lflorea1/projects/RoadMap/splice_list.txt')
ref_file = '/home-2/gyang22@jhu.edu/work/shared/genomes/hg38/hg38.fa'
base_dir = Path('/home-2/gyang22@jhu.edu/methylation-splicing-project')
work_dir = base_dir / 'work_dir_12_tissues'
model_dir = Path('/home-2/gyang22@jhu.edu/MntJULiP/lib')

splice_sites, splice_site_genes_dict = get_info_from_annotation(gtf_file)
splice_site_introns_dict, intron_counts_dict = process_splice_files(splice_list_file, splice_sites)
train_splice_site_introns_dict, test_splice_site_introns_dict, test_genes = train_test_split(splice_site_introns_dict, splice_site_genes_dict)

print(len(train_splice_site_introns_dict), len(test_splice_site_introns_dict))

chr_names = ['chr{}'.format(i) for i in range(1, 23)] + ['chrX', 'chrY']
ref_seqs = get_reference_sequence(ref_file, chr_names)

train_input_dict = generate_neural_network_inputs(train_splice_site_introns_dict, intron_counts_dict, ref_seqs)
test_input_dict = generate_neural_network_inputs(test_splice_site_introns_dict, intron_counts_dict, ref_seqs)

batch_size = 256
data_sets = []
for input_dict in [train_input_dict, test_input_dict]:
    _list = []
    for size, inputs_list in input_dict.items():
        actual_batch_size = batch_size // size
        _length = len(inputs_list)
        for i in range(0, _length, actual_batch_size):
            _list.append(inputs_list[i: min(i + actual_batch_size, _length)])
    data_sets.append(_list)

pkl_file = work_dir / 'data.pkl'
pickle.dump([data_sets[0], data_sets[1]], open(pkl_file, "wb"))


conditions = get_conditions()
test_groups_batches, test_results_batches, test_coords_batches = mntjulip_DM_model_processing(model_dir, test_splice_site_introns_dict, intron_counts_dict, conditions)

pkl_file = work_dir / 'data_mntjulip.pkl'

pickle.dump([test_splice_site_introns_dict, conditions, test_groups_batches,
            test_results_batches, test_coords_batches, test_genes], open(pkl_file, "wb"))
