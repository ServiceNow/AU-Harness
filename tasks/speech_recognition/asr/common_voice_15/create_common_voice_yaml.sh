#!/bin/bash

# List of language codes from asr.json for common_voice_15
languages=(
  "ar" "as" "ast" "az" "ba" "bas" "be" "bg" "bn" "br" "ca" "ckb" "cnh" "cs" 
  "cv" "cy" "da" "de" "dv" "dyu" "el" "en" "eo" "es" "et" "eu" "fa" "fi" 
  "fr" "fy-NL" "ga-IE" "gl" "ha" "hi" "hsb" "hu" "hy-AM" "ia" "id" "ig" 
  "it" "ja" "ka" "kab" "kk" "kmr" "ko" "ky" "lg" "lt" "lv" "mdf" "mg" 
  "mk" "ml" "mn" "mr" "mt" "myv" "ne-NP" "nl" "nn-NO" "or" "pa-IN" "pl" 
  "pt" "rm-sursilv" "rm-vallader" "ro" "ru" "rw" "sah" "sat" "sc" "sk" "sl" 
  "sr" "sv-SE" "sw" "ta" "te" "tg" "th" "ti" "tok" "tr" "tt" "ug" "uk" 
  "ur" "uz" "vot" "yi" "yue" "zh-CN" "zh-HK" "zh-TW"
)

# Directory where the YAML files will be created
dir="/mnt/core_llm_large/akshay/LALMEval/runspecs/speech_recognition/asr/common_voice_15"

# Create YAML files for each language
for lang in "${languages[@]}"; do
  # Skip languages we've already created
  if [ "$lang" == "ab" ] || [ "$lang" == "af" ] || [ "$lang" == "am" ]; then
    continue
  fi
  
  filename="${dir}/common_voice_15_${lang}.yaml"
  
  cat > "$filename" << EOF
task_name: common_voice_15_${lang}
extends: ["../base.yaml#"]
subset: ${lang}
language: ${lang}
EOF

  echo "Created $filename"
done

echo "All common_voice_15 YAML files have been created."
