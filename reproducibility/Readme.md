# Explanation

This is to setup a Neo4J instance for computing datasets for later use.
This is so that we ensure we all have the same versions and things are repeatable across environments.
We dont want the "works in my computer" problem here.

## Preparation

Ensure the following files are in the root of this directory:

### Dependencies

1. Decrypt your documents: Just run `sh ./decrypt.sh`. Ask admin for password if you dont have it
    Make sure these new documents **are NOT** submitted to the github. These are extremely sensitive. Theres already a `.gitignore` with those rules there but just be safe.
2. Install python environment. Either via poetry or simple environment:
   - Via Poetry: `poetry install && poetry shell`
   - Via python:
     - `python -m venv ./some/directory/that/will/not/interfere/with/repo`
     - `source ./some/directory/that/will/not/interfere/with/repo`
     - `pip install -r ./requirements.txt`
3. Ensure the gcloud ansible galaxy is installed.
   - `ansible-galaxy collection install google.cloud`.
     You can confirm installation by checking the output of `ansible-galaxy collection list`
     See [gcloud collection](https://docs.ansible.com/ansible/latest/collections/google/cloud/gcp_compute_inventory.html#ansible-collections-google-cloud-gcp-compute-inventory-requirements) for more info

### Authentication

1. Install [Glcoud CLI Tool](https://cloud.google.com/sdk/docs/install)
2. Authenticate with an account that has access to the cloud console: 
    ```shell
    gcloud auth login
    ```
3. Ensure proper login via:
    ```shell
    gcloud auth list
    ```
    You should see your account appearing there. It should appear as active.
    As FYI: This is needed so that you get the keys to access the google virtual machine.
4. Add your public key to the instance:
    ```
    gcloud compute instances add-metadata INSTANCE_NAME \
    --zone ZONE_NAME \
    --metadata ssh-keys="USERNAME:$(cat ~/.ssh/your-key-name.pub)"
    ```


## Steps to run:

1. Ensure your instance is running. Go to cloud [compute console](htpps://console.cloud.google.com/compute/)
Run 

```shell
ansible-playbook -i gcp_compute.yaml setup_neo4j_poetry.yml
```

Thats it. The shell is up to date and running with whatever you need.
