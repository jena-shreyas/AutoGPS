#!/bin/bash

for i in {0..39}
do
    mkdir -p train/${i}
    cp data.json train/${i}/data.json
done

for i in {0..9}
do
    mkdir -p test/${i}
    cp data.json test/${i}/data.json
done
