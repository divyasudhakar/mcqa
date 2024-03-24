setup: requirements.txt
	pip install -r requirements.txt

run:
	python testbench.py

test:
	python -m unittest

clean:
	rm -rf __pycache__
	rm app.log
