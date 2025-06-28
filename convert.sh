#!/bin/bash
cd resume/resume-general/
latexpand luca_resume.tex | pandoc -f latex -t markdown -o ../../content/resume.md
cd ../../