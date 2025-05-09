import pyBigWig
from Bio import SeqIO
import os
import gzip
import glob


def generate_input_files_from_bw(bw_fnames,
                                 output_dir,
                                 chr_fnames,
                                 seq_length=2500,
                                 interval=1250,
                                 method='sum'):
    bw_map = {}
    os.makedirs(output_dir, exist_ok=True)
    output_fasta = open(os.path.join(output_dir, 'sequences.fasta'), "w")
    chr_len = 0
    bw_fnames = glob.glob(bw_fnames)
    for bw_file in bw_fnames:
        sample = bw_file.split('/')[3].split('_')[0]
        bw_map[sample] = bw_file
    bw_0 = pyBigWig.open(list(bw_map.values())[0])
    for chr_id in chr_fnames.keys():
        print(chr_id)
        chr_fname = chr_fnames[chr_id]
        chr_len = bw_0.chroms(chr_id)
        with gzip.open(chr_fname, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                chr_seq = str(record.seq)
                bw_idx = 0
                while bw_idx + seq_length < chr_len:
                    seq_id =  ",".join([chr_id, str(bw_idx), str(bw_idx+seq_length)])
                    seq = chr_seq[bw_idx:bw_idx + seq_length]
                    output_fasta.write(">" + seq_id + "\n" + seq + "\n")
                    for sample, bw_file in bw_map.items():
                        bw = pyBigWig.open(bw_file)
                        output_faste = open(os.path.join(output_dir, (sample + '.faste')), "a")
                        coverage = ",".join(map(str, bw.stats(chr_id, bw_idx, bw_idx + seq_length, type=method)))
                        output_faste.write(">" + seq_id + "\n" + coverage + "\n")
                    bw_idx += interval


if __name__ == '__main__':
    chr_fnames = {'Chr1': 'Data/RefGenome/Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz',
                'Chr2': 'Data/RefGenome/Arabidopsis_thaliana.TAIR10.dna.chromosome.2.fa.gz',
                'Chr3': 'Data/RefGenome/Arabidopsis_thaliana.TAIR10.dna.chromosome.3.fa.gz',
                'Chr4': 'Data/RefGenome/Arabidopsis_thaliana.TAIR10.dna.chromosome.4.fa.gz',
                'Chr5': 'Data/RefGenome/Arabidopsis_thaliana.TAIR10.dna.chromosome.5.fa.gz'}

    bw_fnames = 'Data/chromatin_cs425/*/*.rpgc.bw'
    output_dir = 'Data/Parsed_Data_Mean'
    generate_input_files_from_bw(bw_fnames,
                                output_dir,
                                chr_fnames,
                                seq_length=2500,
                                interval=2500)






















