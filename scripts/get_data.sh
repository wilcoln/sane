# Create data directory if not exists
mkdir -p data

# Download ESNLI
mkdir -p data/esnli
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv --output data/esnli/esnli_train_1.csv
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_2.csv --output data/esnli/esnli_train_2.csv
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv --output data/esnli/esnli_test.csv
curl https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_dev.csv --output data/esnli/esnli_dev.csv