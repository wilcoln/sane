# Create data directory if not exists
mkdir -p data

# Download ESNLI
mkdir -p data/esnli
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv --output data/esnli/esnli_train_1.csv
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_2.csv --output data/esnli/esnli_train_2.csv
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv --output data/esnli/esnli_test.csv
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_dev.csv --output data/esnli/esnli_dev.csv

# Download ConceptNet
mkdir -p data/conceptnet
wget -nc -P data/conceptnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd data/conceptnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
# download ConceptNet entity embedding
wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../