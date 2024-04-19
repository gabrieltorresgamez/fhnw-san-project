.PHONY: help reqs dl-data

## Show this help message
help:
	@echo ""
	@echo "Usage: make [target]"
	@echo "---"
	@echo "reqs"
	@echo "		Install Python Dependencies"
	@echo "dl-data"
	@echo "		Download Offshore Leaks Dataset"
	@echo ""

## Install Python Dependencies
reqs:
	pip3 install -r requirements.txt --force-reinstall

## Download Offshore Leaks Dataset
dl-data:
	mkdir -p data
	curl https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip --output data/download.zip
	unzip data/download.zip -d data/
	rm data/download.zip

# Set the default goal to `help`
.DEFAULT_GOAL := help