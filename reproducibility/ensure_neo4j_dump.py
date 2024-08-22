#!/usr/bin/python

from ansible.module_utils.basic import AnsibleModule
import subprocess

def run_neo4j_import(dump_file, neo4j_home):
    # Command to check if the database already contains the dump
    check_command = [
        neo4j_home + "/bin/neo4j-admin", "database", "load" ,
        f"--from-path={dump_file}",
        "--database=neo4j",
        "--force"
    ]

    result = subprocess.run(check_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        return False, result.stderr.decode('utf-8')
    return True, result.stdout.decode('utf-8')

def main():
    module_args = dict(
        dump_file=dict(type='str', required=True),
        neo4j_home=dict(type='str', required=True),
    )

    result = dict(
        changed=False,
        message=''
    )

    module = AnsibleModule(
        argument_spec=module_args,
        supports_check_mode=True
    )

    dump_file = module.params['dump_file']
    neo4j_home = module.params['neo4j_home']

    success, output = run_neo4j_import(dump_file, neo4j_home)
    if not success:
        module.fail_json(msg=output)

    result['changed'] = True
    result['message'] = output

    module.exit_json(**result)

if __name__ == '__main__':
    main()
