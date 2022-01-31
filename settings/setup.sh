
echo "Installing PythonVEnv"
sudo apt install -y python3-venv 

echo "Setting ISENV"
python3.6 -m venv isenv

printf "\nexport ATCISELWORKDIR=`dirname $PWD`" >> isenv/bin/activate

source isenv/bin/activate

echo "ATCISELWORKDIR = ${ATCISELWORKDIR}"

pip install --upgrade pip wheel setuptools

pip install -r is_requirements.txt

deactivate

echo "Setting DBENV"
python3.6 -m venv dbenv

printf "\nexport ATCISELWORKDIR=`dirname $PWD`" >> dbenv/bin/activate

source dbenv/bin/activate

pip install --upgrade pip wheel setuptools

pip install -r deep_requirements.txt

deactivate


#install R e 
#install.packages("stats")