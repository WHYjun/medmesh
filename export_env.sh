#!/bin/bash

source test.env
export $(cut -d= -f1 .env)