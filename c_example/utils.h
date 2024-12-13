#ifndef UTILS_H
#define UTILS_H

#include "tcn.h"

/**
 * Carica pesi e bias dai file (o da altre sorgenti) nel modello TCN.
 * Qui presumiamo formati semplici: ad esempio file binari con float in ordine noto.
 * In un caso reale, servirebbe parsare i pesi esportati da PyTorch.
 */
int load_tcn_weights(TCNModel *model, const char *weights_dir);

#endif
