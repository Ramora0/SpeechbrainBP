# Cluster Setup — Pick up from here

## What's done
- Repo pushed to GitHub with V100 slurm script and setup script
- Repo was accidentally cloned to scratch — needs to be moved

## Steps remaining

### 1. Remove the mis-placed clone
```bash
rm -rf /fs/scratch/PAS2836/lees_stuff/librispeechbrain/SpeechbrainBP
```

### 2. Clone to the correct location
```bash
cd ~/research
git clone https://github.com/Ramora0/SpeechbrainBP.git
```

### 3. Run the setup script (creates .v100 venv, installs SpeechBrain from ~/research/speechbrain)
```bash
cd ~/research/SpeechbrainBP
bash slurms/setup-v100.sh
```

### 4. Verify and submit a test run
```bash
cd ~/research/SpeechbrainBP
source .v100/bin/activate
python -c "import speechbrain; print('OK')"
sbatch slurms/train-v100.slurm
```

## Delete this file once setup is complete.
