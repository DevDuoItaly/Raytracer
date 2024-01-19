#!/bin/bash

sudo -u postgres psql -c "CREATE DATABASE imagedb;" -c "CREATE USER root;" -c "ALTER USER root WITH PASSWORD 'password';"
