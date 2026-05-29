# AutoFeat augmentation pipeline — PoC convenience targets.
# Every target is a thin wrapper that prints what it's doing first.

CONDA_ENV   ?= autofeat-6g
PYTHON      := conda run --no-capture-output -n $(CONDA_ENV) python
STREAMLIT   := conda run --no-capture-output -n $(CONDA_ENV) streamlit

.PHONY: help setup neo4j demo smoke dashboard summary clean reset-graph

help:
	@echo "PoC targets:"
	@echo "  make setup       — verify Python env, deps, and Neo4j connectivity"
	@echo "  make neo4j       — start the Neo4j container (docker-compose)"
	@echo "  make demo        — run the two showcase scenarios (2C + KUL) end-to-end"
	@echo "  make smoke       — run smoke tests; fails non-zero on R²/accuracy regression"
	@echo "  make summary     — regenerate results/6g_data/SUMMARY.md from saved runs"
	@echo "  make dashboard   — launch the Streamlit dashboard on port 8501"
	@echo "  make reset-graph — wipe the Neo4j graph (use between unrelated runs)"
	@echo "  make clean       — remove auto_pipeline_* result files (keep historical)"

setup:
	@echo ">>> checking conda env [$(CONDA_ENV)] ..."
	@conda env list | grep -q $(CONDA_ENV) || (echo "env not found; create with: conda create -n $(CONDA_ENV) python=3.10" && exit 1)
	@echo ">>> checking core imports ..."
	@$(PYTHON) -c "import autogluon, sentence_transformers, neo4j, polars, xxhash; \
		print(f'autogluon={autogluon.__version__ if hasattr(autogluon,\"__version__\") else \"?\"} | sentence_transformers={sentence_transformers.__version__} | neo4j={neo4j.__version__} ok')"
	@echo ">>> checking Neo4j reachability ..."
	@$(PYTHON) -c "from neo4j import GraphDatabase; \
		d=GraphDatabase.driver('bolt://localhost:7687', auth=None); \
		s=d.session(); print('nodes:', s.run('MATCH (n) RETURN count(n) AS c').single()['c']); s.close(); d.close()" \
		|| (echo '>>> Neo4j not reachable; run: make neo4j' && exit 1)
	@echo ">>> environment ready."

neo4j:
	@echo ">>> starting Neo4j (docker-compose) ..."
	docker-compose up -d neo4j
	@echo ">>> waiting for bolt on 7687 ..."
	@until lsof -nP -iTCP:7687 -sTCP:LISTEN >/dev/null 2>&1; do sleep 2; done
	@echo ">>> Neo4j up. Browser at http://localhost:7474"

reset-graph:
	@echo ">>> wiping Neo4j graph (batched) ..."
	@$(PYTHON) -c "from neo4j import GraphDatabase; \
		d=GraphDatabase.driver('bolt://localhost:7687', auth=None); s=d.session(); \
		import sys; n=0; \
		[ (lambda r: (sys.stdout.write(f'deleted {r}\\n'), sys.stdout.flush()))(s.run('MATCH ()-[r]->() WITH r LIMIT 50000 DELETE r RETURN count(r) AS c').single()['c']) for _ in range(20) if s.run('MATCH ()-[r]->() RETURN count(r) AS c').single()['c']>0 ]; \
		s.run('MATCH (n) DELETE n').consume(); print('done'); s.close(); d.close()"

demo:
	@echo ">>> [1/2] scenario2c — feature-recovery (self-join via time)"
	$(PYTHON) -m feature_discovery.auto_pipeline \
		--base-table scenarios/scenario2c/rabbitmq-reduced.csv --target lat99 \
		--data-dir scenarios/scenario2c --dataset-type regression \
		--temporal-key time --temporal-tolerance 0 \
		--algorithms XGB --label scenario2c
	@echo
	@echo ">>> [2/2] scenarioK_csi — MaMIMO CSI (binary, 16-antenna lake)"
	$(PYTHON) -m feature_discovery.auto_pipeline \
		--base-table scenarios/scenarioK_csi/samples_base.csv --target target_x \
		--data-dir scenarios/scenarioK_csi --dataset-type binary \
		--no-transformer-discovery \
		--algorithms XGB --label scenarioK_csi
	@echo
	@echo ">>> regenerating summary report ..."
	$(PYTHON) scripts/summarize_results.py
	@echo ">>> demo complete. See results/6g_data/SUMMARY.md"

smoke:
	@echo ">>> running smoke tests (asserts AutoFeat lift on showcase scenarios) ..."
	$(PYTHON) scripts/smoke_test.py

summary:
	$(PYTHON) scripts/summarize_results.py

dashboard:
	@echo ">>> launching dashboard on http://localhost:8501 (Ctrl-C to stop)"
	$(STREAMLIT) run dashboards/augmentation_dashboard.py

clean:
	@echo ">>> removing auto_pipeline_* results (keeps benchmark_6g_* and kul_*) ..."
	rm -f results/6g_data/auto_pipeline_*.csv
	rm -f results/6g_data/SUMMARY.md results/6g_data/summary.csv
