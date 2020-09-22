import os
import click

from modules.sparse_flow import calc_flow
from modules.dense_flow import calc_flow as calc_flow_dense
from modules.util import get_config



@click.command()
@click.option('--config', help="path of config file")
def sparse(config):
  configs = get_config(config)
  calc_flow(configs)

@click.command()
@click.option('--config', help="path of config file")
def dense(config):
  configs = get_config(config)
  calc_flow_dense(configs)


@click.group()
def main():
  pass

if __name__ == '__main__':
  main.add_command(sparse)
  main.add_command(dense)
  main()