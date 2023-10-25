#!/bin/bash

for i in {1..6}; do ./solve-daily-quizz.sh >>/mnt/d/Desktop/Ractinver/Scripts/slog.txt && ./get-daily-quizz-answers.sh >> /mnt/d/Desktop/Ractinver/Scripts/slog.txt ; sleep 3 ; done
