# Reproducible build for figures, models, and contributions
# Usage examples:
#   make all
#   make figures
#   make train
#   make contributions
#   make clean-figs
#   make clean

PYTHON ?= python3
SCRIPT_DIR := script
DATA_FILE := data/deidentified_data.csv
RESULT_DIR := result

# --- helper ---

.PHONY: all setup check-data check-dirs requirements figures stat1 stat2 pca_logreg train contributions clean-figs clean

all: setup figures train contributions

setup: check-data check-dirs

check-data:
	@test -f $(DATA_FILE) || (echo "[ERROR] Missing $(DATA_FILE). Please place the dataset first." && exit 1)

check-dirs:
	@mkdir -p $(RESULT_DIR)

requirements:
	$(PYTHON) -m pip install -r requirements.txt

# --- figures ---

figures: stat1 stat2 pca_logreg

stat1:
	@echo "[RUN] statistic_analysis1.py"
	@cd $(SCRIPT_DIR) && $(PYTHON) statistic_analysis1.py

stat2:
	@echo "[RUN] statistic_analysis2.py"
	@cd $(SCRIPT_DIR) && $(PYTHON) statistic_analysis2.py

pca_logreg:
	@echo "[RUN] PCA_LogisticRegression.py"
	@cd $(SCRIPT_DIR) && $(PYTHON) PCA_LogisticRegression.py

# --- models & contributions ---

train:
	@echo "[RUN] model_training_xgboost.py"
	@cd $(SCRIPT_DIR) && $(PYTHON) model_training_xgboost.py

contributions:
	@echo "[RUN] feature_contribution.py"
	@cd $(SCRIPT_DIR) && $(PYTHON) feature_contribution.py

# --- cleaning ---

clean-figs:
	@echo "[CLEAN] Removing generated figures"
	@rm -f $(RESULT_DIR)/*.pdf $(RESULT_DIR)/*.png $(RESULT_DIR)/*.svg

clean: clean-figs
	@echo "[CLEAN] Removing model artifacts"
	@rm -f $(RESULT_DIR)/*.json $(RESULT_DIR)/*.txt
