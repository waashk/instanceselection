
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


