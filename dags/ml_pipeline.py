from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_credit_risk_pipeline',
    default_args=default_args,
    description='Monthly credit risk model training, inference, and monitoring pipeline',
    schedule='0 0 1 * *',  # Monthly
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31),
    catchup=True,
) as dag:

    # --- Model Training ---
    model_train = BashOperator(
        task_id='model_train',
        bash_command='python3 /opt/airflow/scripts/model_train.py --snapshotdate "{{ ds }}"',
        cwd='/opt/airflow',
    )

    # --- Model Inference ---
    model_inference = BashOperator(
        task_id='model_inference',
        bash_command='python3 /opt/airflow/scripts/model_inference.py --snapshotdate "{{ ds }}"',
        cwd='/opt/airflow',
    )

    # --- Model Monitoring & Drift Detection ---
    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command='python3 /opt/airflow/scripts/model_monitor.py --snapshotdate "{{ ds }}"',
        cwd='/opt/airflow',
    )

    # Define Dependencies
    model_train >> model_inference >> model_monitor
