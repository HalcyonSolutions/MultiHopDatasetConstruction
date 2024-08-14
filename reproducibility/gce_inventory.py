#!/usr/bin/env python3
import subprocess
import json
import os

def get_gce_ip():
    command = "gcloud compute instances list --format=json"
    if not os.environ.get("OCACHE"):
        output = subprocess.check_output(command, shell=True)
    else:
        print("Using Cached Instances")
        output = str(os.environ.get("OCACHE"))
    instances = json.loads(output)
    host_vars = {}

    for instance in instances:
        name = instance['name']
        network = instance.get('networkInterfaces', [])
        if network:
            ip = network[0].get('accessConfigs', [{}])[0].get('natIP')
            if ip:
                host_vars[name] = {'ansible_host': ip}

    # Let the  usr select only one of the instances
    # Multiple machiens is a project far in the future.
    if len(host_vars) > 1:
        print("There are multiple instances. Please select one:")
        for i, instance in enumerate(host_vars):
            print(f"{i+1}. {instance}")
        choice = input("Enter the number of the instance: ")
        if choice.isdigit():
            choice = int(choice) - 1
            if choice < len(host_vars):
                host_vars = host_vars[choice]
            else:
                print("Invalid choice. Please try again.")
                exit(1)
    else:
        host_vars = host_vars

    print(f"Host vars are {host_vars}")
    return host_vars

def main():
    inventory = {
        'all': {
            'hosts': list(get_gce_ip().keys())
        },
        '_meta': {
            'hostvars': get_gce_ip()
        }
    }

    print(json.dumps(inventory, indent=2))

if __name__ == "__main__":
    main()
