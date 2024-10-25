#ifndef __stepping_H
#define __stepping_H	 
#include "sys.h"
void stepping_init(void);
void stepping_PWM(uint32_t angle,uint32_t speed);
void stepping_reversal(void);
void stepping_End(void);

#endif
