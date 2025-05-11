import os
import requests
import zipfile


def download_reference_genome():
    url = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz"
    output_file = "Arabidopsis_thaliana_reference_genome.fa.gz"

    response = requests.get(url, stream=True)
    chr_fnames = {
        'Chr1': 'Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz',
        'Chr2': 'Arabidopsis_thaliana.TAIR10.dna.chromosome.2.fa.gz',
        'Chr3': 'Arabidopsis_thaliana.TAIR10.dna.chromosome.3.fa.gz',
        'Chr4': 'Arabidopsis_thaliana.TAIR10.dna.chromosome.4.fa.gz',
        'Chr5': 'Arabidopsis_thaliana.TAIR10.dna.chromosome.5.fa.gz'
    }

    os.makedirs("Data/RefGenome", exist_ok=True)

    for chr_name, file_path in chr_fnames.items():
        url = f"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/dna/{os.path.basename(file_path)}"
        output_file = os.path.join("Data/RefGenome", os.path.basename(file_path))

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"{chr_name} downloaded successfully as {output_file}")
        else:
            print(f"Failed to download {chr_name}. HTTP Status Code: {response.status_code}")


def download_accessiblility_data():
    url = "https://zenodo.org/record/10946767/files/chromatin_cs425.zip"
    output_file = "chromatin_cs425.zip"
    output_dir = f"Data"

    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Downloaded {output_file}")

        # Unzip the file
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Unpacked {output_file} to {output_dir}")

        # Remove the zip file after extraction
        os.remove(output_file)
    else:
        print(f"Failed to download chromatin data. HTTP Status Code: {response.status_code}")




if __name__ == '__main__':
    download_reference_genome()
    download_accessiblility_data()






















