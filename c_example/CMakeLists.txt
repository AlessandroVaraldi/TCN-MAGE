cmake_minimum_required(VERSION 3.10)

# Nome del progetto
project(TCNInference C)

# Imposta lo standard C
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Aggiunge i file sorgente
# Puoi elencarli manualmente, oppure usare file(GLOB).
# Qui li scriviamo manualmente per chiarezza.
set(SOURCES
    main.c
    tcn.c
    causal_conv1d.c
    utils.c
)

# Includi la directory corrente per i file header
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Crea l'eseguibile
add_executable(tcn_inference ${SOURCES})

# Opzionalmente, puoi aggiungere flag di compilazione
# ad esempio ottimizzazione o debug
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall")

# Non sono necessarie librerie esterne in questo esempio,
# ma potresti linkarne qualcuna se necessario.
