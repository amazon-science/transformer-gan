###  TransformerXL
/opt/conda/bin/python -m pip install --upgrade pip
python3 -m pip install --user torch==1.5.*
python3 -m pip install --user pypianoroll==0.5.3
python3 -m pip install --user nltk
python3 -m pip install --user texttable
python3 -m pip install --user yacs
git clone https://github.com/NVIDIA/apex.git
cd apex
python3 -m pip install --user -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
###  BERT
python3 -m pip install --user transformers==2.5.1
python3 -m pip install --user tqdm==4.41.1
python3 -m pip install --user tensorboardX
python3 -m pip install --user note_seq

