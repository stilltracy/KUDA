echo n2427.sed
cat <<EOF >n2427.sed

1,/<code>/        d
/<\/code>/,/<code>/    d
            s|<var>||g
            s|</var>||g
            s|&lt;|<|g
            s|&gt;|>|g
            s|&amp;|\&|g

EOF
echo Makefile
cat <<EOF >Makefile

default : test

n2427.bash : n2427.html
	sed -f n2427.sed n2427.html > n2427.bash

stdatomic.h cstdatomic impatomic.h impatomic.c n2427.c : n2427.bash
	bash n2427.bash

impatomic.o : impatomic.h impatomic.c
	gcc -std=c99 -c impatomic.c

n2427.c.exe : n2427.c stdatomic.h impatomic.o
	gcc -std=c99 -o n2427.c.exe n2427.c impatomic.o

n2427.c++.exe : n2427.c stdatomic.h impatomic.o
	g++ -o n2427.c++.exe n2427.c impatomic.o

test : n2427.c.exe n2427.c++.exe

clean :
	rm -f n2427.bash stdatomic.h cstdatomic impatomic.h impatomic.c
	rm -f impatomic.o n2427.c.exe n2427.c++.exe

EOF
echo impatomic.h includes
cat <<EOF >impatomic.h

#ifdef __cplusplus
#include <cstddef>
namespace std {
#else
#include <stddef.h>
#include <stdbool.h>
#endif

EOF
echo impatomic.c includes
cat <<EOF >impatomic.c

#include <stdint.h>
#include "impatomic.h"

EOF
echo impatomic.h CPP0X
cat <<EOF >>impatomic.h

#define CPP0X( feature )

EOF
echo impatomic.h order
cat <<EOF >>impatomic.h

typedef enum memory_order {
    memory_order_relaxed, memory_order_acquire, memory_order_release,
    memory_order_acq_rel, memory_order_seq_cst
} memory_order;

EOF
echo impatomic.h flag
cat <<EOF >>impatomic.h

typedef struct atomic_flag
{
#ifdef __cplusplus
    bool test_and_set( memory_order = memory_order_seq_cst ) volatile;
    void clear( memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;

    CPP0X( atomic_flag() = default; )
    CPP0X( atomic_flag( const atomic_flag& ) = delete; )
    atomic_flag& operator =( const atomic_flag& ) CPP0X(=delete);

CPP0X(private:)
#endif
    bool __f__;
} atomic_flag;

#define ATOMIC_FLAG_INIT { false }

#ifdef __cplusplus
extern "C" {
#endif

extern bool atomic_flag_test_and_set( volatile atomic_flag* );
extern bool atomic_flag_test_and_set_explicit
( volatile atomic_flag*, memory_order );
extern void atomic_flag_clear( volatile atomic_flag* );
extern void atomic_flag_clear_explicit
( volatile atomic_flag*, memory_order );
extern void atomic_flag_fence
( const volatile atomic_flag*, memory_order );
extern void __atomic_flag_wait__
( volatile atomic_flag* );
extern void __atomic_flag_wait_explicit__
( volatile atomic_flag*, memory_order );
extern volatile atomic_flag* __atomic_flag_for_address__
( const volatile void* __z__ )
__attribute__((const));

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

inline bool atomic_flag::test_and_set( memory_order __x__ ) volatile
{ return atomic_flag_test_and_set_explicit( this, __x__ ); }

inline void atomic_flag::clear( memory_order __x__ ) volatile
{ atomic_flag_clear_explicit( this, __x__ ); }

inline void atomic_flag::fence( memory_order __x__ ) const volatile
{ atomic_flag_fence( this, __x__ ); }

#endif

EOF
echo impatomic.c flag
cat <<EOF >>impatomic.c

#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 0)
#define USE_SYNC
#endif
#endif

bool atomic_flag_test_and_set_explicit
( volatile atomic_flag* __a__, memory_order __x__ )
{
#ifdef USE_SYNC
    if ( __x__ >= memory_order_acq_rel )
        __sync_synchronize();
    return __sync_lock_test_and_set( &(__a__->__f__), 1 );
#else
    bool result = __a__->__f__;
    __a__->__f__ = true;
    return result;
#endif
}

bool atomic_flag_test_and_set( volatile atomic_flag* __a__ )
{ return atomic_flag_test_and_set_explicit( __a__, memory_order_seq_cst ); }

void atomic_flag_clear_explicit
( volatile atomic_flag* __a__, memory_order __x__ )
{
#ifdef USE_SYNC
    __sync_lock_release( &(__a__->__f__) );
    if ( __x__ >= memory_order_acq_rel )
        __sync_synchronize();
#else
    __a__->__f__ = false;
#endif
} 

void atomic_flag_clear( volatile atomic_flag* __a__ )
{ atomic_flag_clear_explicit( __a__, memory_order_seq_cst ); }

void atomic_flag_fence( const volatile atomic_flag* __a__, memory_order __x__ )
{ 
#ifdef USE_SYNC
    __sync_synchronize();
#endif
} 

void __atomic_flag_wait__( volatile atomic_flag* __a__ )
{ while ( atomic_flag_test_and_set( __a__ ) ); }

void __atomic_flag_wait_explicit__( volatile atomic_flag* __a__,
                                    memory_order __x__ )
{ while ( atomic_flag_test_and_set_explicit( __a__, __x__ ) ); }

#define LOGSIZE 4

static atomic_flag volatile __atomic_flag_anon_table__[ 1 << LOGSIZE ] =
{
    ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT,
    ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT,
    ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT,
    ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT, ATOMIC_FLAG_INIT,
};

volatile atomic_flag* __atomic_flag_for_address__( const volatile void* __z__ )
{
    uintptr_t __u__ = (uintptr_t)__z__;
    __u__ += (__u__ >> 2) + (__u__ << 4);
    __u__ += (__u__ >> 7) + (__u__ << 5);
    __u__ += (__u__ >> 17) + (__u__ << 13);
    if ( sizeof(uintptr_t) > 4 ) __u__ += (__u__ >> 31);
    __u__ &= ~((~(uintptr_t)0) << LOGSIZE);
    return __atomic_flag_anon_table__ + __u__;
}

EOF
echo impatomic.h macros implementation
cat <<EOF >>impatomic.h

#define _ATOMIC_LOAD_( __a__, __x__ ) \\
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \\
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \\
   __atomic_flag_wait_explicit__( __g__, __x__ ); \\
   __typeof__((__a__)->__f__) __r__ = *__p__; \\
   atomic_flag_clear_explicit( __g__, __x__ ); \\
   __r__; })

#define _ATOMIC_STORE_( __a__, __m__, __x__ ) \\
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \\
   __typeof__(__m__) __v__ = (__m__); \\
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \\
   __atomic_flag_wait_explicit__( __g__, __x__ ); \\
   *__p__ = __v__; \\
   atomic_flag_clear_explicit( __g__, __x__ ); \\
   __v__; })

#define _ATOMIC_MODIFY_( __a__, __o__, __m__, __x__ ) \\
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \\
   __typeof__(__m__) __v__ = (__m__); \\
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \\
   __atomic_flag_wait_explicit__( __g__, __x__ ); \\
   __typeof__((__a__)->__f__) __r__ = *__p__; \\
   *__p__ __o__ __v__; \\
   atomic_flag_clear_explicit( __g__, __x__ ); \\
   __r__; })

#define _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ) \\
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \\
   __typeof__(__e__) __q__ = (__e__); \\
   __typeof__(__m__) __v__ = (__m__); \\
   bool __r__; \\
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \\
   __atomic_flag_wait_explicit__( __g__, __x__ ); \\
   __typeof__((__a__)->__f__) __t__ = *__p__; \\
   if ( __t__ == *__q__ ) { *__p__ = __v__; __r__ = true; } \\
   else { *__q__ = __t__; __r__ = false; } \\
   atomic_flag_clear_explicit( __g__, __x__ ); \\
   __r__; })

#define _ATOMIC_FENCE_( __a__, __x__ ) \\
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \\
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \\
   atomic_flag_fence( __g__, __x__ ); \\
   })

EOF
echo impatomic.h lock-free macros
cat <<EOF >>impatomic.h

#define ATOMIC_INTEGRAL_LOCK_FREE 0
#define ATOMIC_ADDRESS_LOCK_FREE 0

EOF
bool="bool"
address="void*"

INTEGERS="char schar uchar short ushort int uint long ulong llong ullong"
char="char"
schar="signed char"
uchar="unsigned char"
short="short"
ushort="unsigned short"
int="int"
uint="unsigned int"
long="long"
ulong="unsigned long"
llong="long long"
ullong="unsigned long long"

CHARACTERS="wchar_t"
# CHARACTERS="char16_t char32_t wchar_t" // char*_t not yet in compilers
char16_t="char16_t"
char32_t="char32_t"
wchar_t="wchar_t"
ADR_OPERATIONS="add sub"
INT_OPERATIONS="add sub and or xor"
add="+"
sub="-"
and="&"
or="|"
xor="^"
echo impatomic.h type boolean
cat <<EOF >>impatomic.h

typedef struct atomic_bool
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( bool, memory_order = memory_order_seq_cst ) volatile;
    bool load( memory_order = memory_order_seq_cst ) volatile;
    bool swap( bool, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap ( bool&, bool, memory_order, memory_order ) volatile;
    bool compare_swap ( bool&, bool,
                        memory_order = memory_order_seq_cst) volatile;
    void fence( memory_order ) const volatile;

    CPP0X( atomic_bool() = delete; )
    CPP0X( constexpr explicit atomic_bool( bool __v__ ) : __f__( __v__ ) { } )
    CPP0X( atomic_bool( const atomic_bool& ) = delete; )
    atomic_bool& operator =( const atomic_bool& ) CPP0X(=delete);

    bool operator =( bool __v__ ) volatile
    { store( __v__ ); return __v__; }

    friend void atomic_store_explicit( volatile atomic_bool*, bool,
                                       memory_order );
    friend bool atomic_load_explicit( volatile atomic_bool*, memory_order );
    friend bool atomic_swap_explicit( volatile atomic_bool*, bool,
                                      memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_bool*, bool*, bool,
                                              memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_bool*, memory_order );

CPP0X(private:)
#endif
    bool __f__;
} atomic_bool;

EOF
echo impatomic.h type address
cat <<EOF >>impatomic.h

typedef struct atomic_address
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( void*, memory_order = memory_order_seq_cst ) volatile;
    void* load( memory_order = memory_order_seq_cst ) volatile;
    void* swap( void*, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( void*&, void*, memory_order, memory_order ) volatile;
    bool compare_swap( void*&, void*,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    void* fetch_add( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;
    void* fetch_sub( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_address() = default; )
    CPP0X( constexpr explicit atomic_address( void* __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_address( const atomic_address& ) = delete; )
    atomic_address& operator =( const atomic_address & ) CPP0X(=delete);

    void* operator =( void* __v__ ) volatile
    { store( __v__ ); return __v__; }

    void* operator +=( ptrdiff_t __v__ ) volatile
    { return fetch_add( __v__ ); }

    void* operator -=( ptrdiff_t __v__ ) volatile
    { return fetch_sub( __v__ ); }

    friend void atomic_store_explicit( volatile atomic_address*, void*,
                                       memory_order );
    friend void* atomic_load_explicit( volatile atomic_address*, memory_order );
    friend void* atomic_swap_explicit( volatile atomic_address*, void*,
                                       memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_address*,
                              void**, void*, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_address*, memory_order );
    friend void* atomic_fetch_add_explicit( volatile atomic_address*, ptrdiff_t,
                                            memory_order );
    friend void* atomic_fetch_sub_explicit( volatile atomic_address*, ptrdiff_t,
                                            memory_order );

CPP0X(private:)
#endif
    void* __f__;
} atomic_address;

EOF
echo impatomic.h type integers
for TYPEKEY in ${INTEGERS}
do
TYPENAME=${!TYPEKEY}
cat <<EOF >>impatomic.h

typedef struct atomic_${TYPEKEY}
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( ${TYPENAME},
                memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} load( memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} swap( ${TYPENAME},
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( ${TYPENAME}&, ${TYPENAME},
                       memory_order, memory_order ) volatile;
    bool compare_swap( ${TYPENAME}&, ${TYPENAME},
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    ${TYPENAME} fetch_add( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_sub( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_and( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_or( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_xor( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_${TYPEKEY}() = default; )
    CPP0X( constexpr atomic_${TYPEKEY}( ${TYPENAME} __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_${TYPEKEY}( const atomic_${TYPEKEY}& ) = delete; )
    atomic_${TYPEKEY}& operator =( const atomic_${TYPEKEY}& ) CPP0X(=delete);

    ${TYPENAME} operator =( ${TYPENAME} __v__ ) volatile
    { store( __v__ ); return __v__; }

    ${TYPENAME} operator ++( int ) volatile
    { return fetch_add( 1 ); }

    ${TYPENAME} operator --( int ) volatile
    { return fetch_sub( 1 ); }

    ${TYPENAME} operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    ${TYPENAME} operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    ${TYPENAME} operator +=( ${TYPENAME} __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    ${TYPENAME} operator -=( ${TYPENAME} __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    ${TYPENAME} operator &=( ${TYPENAME} __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    ${TYPENAME} operator |=( ${TYPENAME} __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    ${TYPENAME} operator ^=( ${TYPENAME} __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_${TYPEKEY}*, ${TYPENAME},
                                       memory_order );
    friend ${TYPENAME} atomic_load_explicit( volatile atomic_${TYPEKEY}*,
                                             memory_order );
    friend ${TYPENAME} atomic_swap_explicit( volatile atomic_${TYPEKEY}*,
                                             ${TYPENAME}, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_${TYPEKEY}*,
                      ${TYPENAME}*, ${TYPENAME}, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_${TYPEKEY}*, memory_order );
    friend ${TYPENAME} atomic_fetch_add_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_sub_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_and_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_or_explicit(  volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_xor_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );

CPP0X(private:)
#endif
    ${TYPENAME} __f__;
} atomic_${TYPEKEY};

EOF
done
echo impatomic.h typedefs integers
cat <<EOF >>impatomic.h

typedef atomic_schar atomic_int_least8_t;
typedef atomic_uchar atomic_uint_least8_t;
typedef atomic_short atomic_int_least16_t;
typedef atomic_ushort atomic_uint_least16_t;
typedef atomic_int atomic_int_least32_t;
typedef atomic_uint atomic_uint_least32_t;
typedef atomic_llong atomic_int_least64_t;
typedef atomic_ullong atomic_uint_least64_t;

typedef atomic_schar atomic_int_fast8_t;
typedef atomic_uchar atomic_uint_fast8_t;
typedef atomic_short atomic_int_fast16_t;
typedef atomic_ushort atomic_uint_fast16_t;
typedef atomic_int atomic_int_fast32_t;
typedef atomic_uint atomic_uint_fast32_t;
typedef atomic_llong atomic_int_fast64_t;
typedef atomic_ullong atomic_uint_fast64_t;

typedef atomic_long atomic_intptr_t;
typedef atomic_ulong atomic_uintptr_t;

typedef atomic_long atomic_ssize_t;
typedef atomic_ulong atomic_size_t;

typedef atomic_long atomic_ptrdiff_t;

typedef atomic_llong atomic_intmax_t;
typedef atomic_ullong atomic_uintmax_t;

EOF
echo impatomic.h type characters
cat <<EOF >>impatomic.h

#ifdef __cplusplus

EOF

for TYPEKEY in ${CHARACTERS}
do
TYPENAME=${!TYPEKEY}
cat <<EOF >>impatomic.h

typedef struct atomic_${TYPEKEY}
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( ${TYPENAME}, memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} load( memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} swap( ${TYPENAME},
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( ${TYPENAME}&, ${TYPENAME},
                       memory_order, memory_order ) volatile;
    bool compare_swap( ${TYPENAME}&, ${TYPENAME},
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    ${TYPENAME} fetch_add( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_sub( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_and( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_or( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;
    ${TYPENAME} fetch_xor( ${TYPENAME},
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_${TYPENAME}() = default; )
    CPP0X( constexpr atomic_${TYPEKEY}( ${TYPENAME} __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_${TYPENAME}( const atomic_${TYPENAME}& ) = delete; )
    atomic_${TYPENAME}& operator =( const atomic_${TYPENAME}& ) CPP0X(=delete);

    ${TYPENAME} operator =( ${TYPENAME} __v__ ) volatile
    { store( __v__ ); return __v__; }

    ${TYPENAME} operator ++( int ) volatile
    { return fetch_add( 1 ); }

    ${TYPENAME} operator --( int ) volatile
    { return fetch_sub( 1 ); }

    ${TYPENAME} operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    ${TYPENAME} operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    ${TYPENAME} operator +=( ${TYPENAME} __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    ${TYPENAME} operator -=( ${TYPENAME} __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    ${TYPENAME} operator &=( ${TYPENAME} __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    ${TYPENAME} operator |=( ${TYPENAME} __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    ${TYPENAME} operator ^=( ${TYPENAME} __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_${TYPEKEY}*, ${TYPENAME},
                                       memory_order );
    friend ${TYPENAME} atomic_load_explicit( volatile atomic_${TYPEKEY}*,
                                             memory_order );
    friend ${TYPENAME} atomic_swap_explicit( volatile atomic_${TYPEKEY}*,
                                             ${TYPENAME}, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_${TYPEKEY}*,
                    ${TYPENAME}*, ${TYPENAME}, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_${TYPEKEY}*, memory_order );
    friend ${TYPENAME} atomic_fetch_add_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_sub_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_and_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_or_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );
    friend ${TYPENAME} atomic_fetch_xor_explicit( volatile atomic_${TYPEKEY}*,
                                                  ${TYPENAME}, memory_order );

CPP0X(private:)
#endif
    ${TYPENAME} __f__;
} atomic_${TYPEKEY};

EOF
done

cat <<EOF >>impatomic.h

#else

typedef atomic_int_least16_t atomic_char16_t;
typedef atomic_int_least32_t atomic_char32_t;
typedef atomic_int_least32_t atomic_wchar_t;

#endif

EOF
echo impatomic.h type generic
cat <<EOF >>impatomic.h

#ifdef __cplusplus

template< typename T >
struct atomic
{
#ifdef __cplusplus

    bool is_lock_free() const volatile;
    void store( T, memory_order = memory_order_seq_cst ) volatile;
    T load( memory_order = memory_order_seq_cst ) volatile;
    T swap( T __v__, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( T&, T, memory_order, memory_order ) volatile;
    bool compare_swap( T&, T, memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;

    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( T __v__ ) : __f__( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    T operator =( T __v__ ) volatile
    { store( __v__ ); return __v__; }

CPP0X(private:)
#endif
    T __f__;
};

#endif
EOF
echo impatomic.h type pointer
cat <<EOF >>impatomic.h

#ifdef __cplusplus

template<typename T> struct atomic< T* > : atomic_address
{
    T* load( memory_order = memory_order_seq_cst ) volatile;
    T* swap( T*, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( T*&, T*, memory_order, memory_order ) volatile;
    bool compare_swap( T*&, T*,
                       memory_order = memory_order_seq_cst ) volatile;
    T* fetch_add( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;
    T* fetch_sub( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( T __v__ ) : atomic_address( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    T* operator =( T* __v__ ) volatile
    { store( __v__ ); return __v__; }

    T* operator ++( int ) volatile
    { return fetch_add( 1 ); }

    T* operator --( int ) volatile
    { return fetch_sub( 1 ); }

    T* operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    T* operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    T* operator +=( T* __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    T* operator -=( T* __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }
};

#endif
EOF
echo impatomic.h type specializations
cat <<EOF >>impatomic.h

#ifdef __cplusplus

EOF

for TYPEKEY in bool address ${INTEGERS} ${CHARACTERS}
do
TYPENAME=${!TYPEKEY}
cat <<EOF >>impatomic.h

template<> struct atomic< ${TYPENAME} > : atomic_${TYPEKEY}
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( ${TYPENAME} __v__ )
    : atomic_${TYPEKEY}( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    ${TYPENAME} operator =( ${TYPENAME} __v__ ) volatile
    { store( __v__ ); return __v__; }
};

EOF
done

cat <<EOF >>impatomic.h

#endif

EOF
cat <<EOF >>impatomic.h

#ifdef __cplusplus

EOF

echo impatomic.h functions ordinary basic
for TYPEKEY in bool address ${INTEGERS} ${CHARACTERS}
do
TYPENAME=${!TYPEKEY}
cat <<EOF >>impatomic.h

inline bool atomic_is_lock_free( const volatile atomic_${TYPEKEY}* __a__ )
{ return false; }

inline ${TYPENAME} atomic_load_explicit
( volatile atomic_${TYPEKEY}* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline ${TYPENAME} atomic_load( volatile atomic_${TYPEKEY}* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME} __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME} __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline ${TYPENAME} atomic_swap_explicit
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME} __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline ${TYPENAME} atomic_swap
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME} __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME}* __e__, ${TYPENAME} __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME}* __e__, ${TYPENAME} __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_${TYPEKEY}* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }

EOF
done

echo impatomic.h functions address fetch
TYPEKEY=address
TYPENAME=${!TYPEKEY}

for FNKEY in ${ADR_OPERATIONS}
do
OPERATOR=${!FNKEY}

cat <<EOF >>impatomic.h

inline ${TYPENAME} atomic_fetch_${FNKEY}_explicit
( volatile atomic_${TYPEKEY}* __a__, ptrdiff_t __m__, memory_order __x__ )
{ ${TYPENAME} volatile* __p__ = &((__a__)->__f__);
  volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ );
  __atomic_flag_wait_explicit__( __g__, __x__ );
  ${TYPENAME} __r__ = *__p__;
  *__p__ = (${TYPENAME})((char*)(*__p__) ${OPERATOR} __m__);
  atomic_flag_clear_explicit( __g__, __x__ );
  return __r__; }

inline ${TYPENAME} atomic_fetch_${FNKEY}
( volatile atomic_${TYPEKEY}* __a__, ptrdiff_t __m__ )
{ return atomic_fetch_${FNKEY}_explicit( __a__, __m__, memory_order_seq_cst ); }

EOF
done

echo impatomic.h functions integer fetch
for TYPEKEY in ${INTEGERS} ${CHARACTERS}
do
TYPENAME=${!TYPEKEY}

for FNKEY in ${INT_OPERATIONS}
do
OPERATOR=${!FNKEY}

cat <<EOF >>impatomic.h

inline ${TYPENAME} atomic_fetch_${FNKEY}_explicit
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME} __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ${OPERATOR}=, __m__, __x__ ); }

inline ${TYPENAME} atomic_fetch_${FNKEY}
( volatile atomic_${TYPEKEY}* __a__, ${TYPENAME} __m__ )
{ atomic_fetch_${FNKEY}_explicit( __a__, __m__, memory_order_seq_cst ); }

EOF
done
done
cat <<EOF >>impatomic.h

#else

EOF

echo impatomic.h type-generic macros basic
cat <<EOF >>impatomic.h

#define atomic_is_lock_free( __a__ ) \\
false

#define atomic_load( __a__ ) \\
_ATOMIC_LOAD_( __a__, memory_order_seq_cst )

#define atomic_load_explicit( __a__, __x__ ) \\
_ATOMIC_LOAD_( __a__, __x__ )

#define atomic_store( __a__, __m__ ) \\
_ATOMIC_STORE_( __a__, __m__, memory_order_seq_cst )

#define atomic_store_explicit( __a__, __m__, __x__ ) \\
_ATOMIC_STORE_( __a__, __m__, __x__ )

#define atomic_swap( __a__, __m__ ) \\
_ATOMIC_MODIFY_( __a__, =, __m__, memory_order_seq_cst )

#define atomic_swap_explicit( __a__, __m__, __x__ ) \\
_ATOMIC_MODIFY_( __a__, =, __m__, __x__ )

#define atomic_compare_swap( __a__, __e__, __m__ ) \\
_ATOMIC_CMPSWP_( __a__, __e__, __m__, memory_order_seq_cst )

#define atomic_compare_swap_explicit( __a__, __e__, __m__, __x__, __y__ ) \\
_ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ )

#define atomic_fence( __a__, __x__ ) \\
({ _ATOMIC_FENCE_( __a__, __x__ ); })

EOF

echo impatomic.h type-generic macros fetch
for FNKEY in ${INT_OPERATIONS}
do
OPERATOR=${!FNKEY}

cat <<EOF >>impatomic.h

#define atomic_fetch_${FNKEY}_explicit( __a__, __m__, __x__ ) \\
_ATOMIC_MODIFY_( __a__, ${OPERATOR}=, __m__, __x__ )

#define atomic_fetch_${FNKEY}( __a__, __m__ ) \\
_ATOMIC_MODIFY_( __a__, ${OPERATOR}=, __m__, memory_order_seq_cst )

EOF
done

cat <<EOF >>impatomic.h

#endif

EOF
cat <<EOF >>impatomic.h

#ifdef __cplusplus

EOF

echo impatomic.h methods ordinary basic
for TYPEKEY in bool address ${INTEGERS} ${CHARACTERS}
do
TYPENAME=${!TYPEKEY}

cat <<EOF >>impatomic.h

inline bool atomic_${TYPEKEY}::is_lock_free() const volatile
{ return false; }

inline void atomic_${TYPEKEY}::store
( ${TYPENAME} __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline ${TYPENAME} atomic_${TYPEKEY}::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline ${TYPENAME} atomic_${TYPEKEY}::swap
( ${TYPENAME} __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_${TYPEKEY}::compare_swap
( ${TYPENAME}& __e__, ${TYPENAME} __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_${TYPEKEY}::compare_swap
( ${TYPENAME}& __e__, ${TYPENAME} __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_${TYPEKEY}::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }

EOF
done

echo impatomic.h methods template basic
cat <<EOF >>impatomic.h

template< typename T >
inline bool atomic<T>::is_lock_free() const volatile
{ return false; }

template< typename T >
inline void atomic<T>::store( T __v__, memory_order __x__ ) volatile
{ _ATOMIC_STORE_( this, __v__, __x__ ); }

template< typename T >
inline T atomic<T>::load( memory_order __x__ ) volatile
{ return _ATOMIC_LOAD_( this, __x__ ); }

template< typename T >
inline T atomic<T>::swap( T __v__, memory_order __x__ ) volatile
{ return _ATOMIC_MODIFY_( this, =, __v__, __x__ ); }

template< typename T >
inline bool atomic<T>::compare_swap
( T& __r__, T __v__, memory_order __x__, memory_order __y__ ) volatile
{ return _ATOMIC_CMPSWP_( this, &__r__, __v__, __x__ ); }

template< typename T >
inline bool atomic<T>::compare_swap
( T& __r__, T __v__, memory_order __x__ ) volatile
{ return compare_swap( __r__, __v__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

EOF

echo impatomic.h methods address fetch
TYPEKEY=address
TYPENAME=${!TYPEKEY}

cat <<EOF >>impatomic.h

inline void* atomic_address::fetch_add
( ptrdiff_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }

inline void* atomic_address::fetch_sub
( ptrdiff_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }

EOF

echo impatomic.h methods integer fetch
for TYPEKEY in ${INTEGERS} ${CHARACTERS}
do
TYPENAME=${!TYPEKEY}

for FNKEY in ${INT_OPERATIONS}
do
OPERATOR=${!FNKEY}

cat <<EOF >>impatomic.h

inline ${TYPENAME} atomic_${TYPEKEY}::fetch_${FNKEY}
( ${TYPENAME} __m__, memory_order __x__ ) volatile
{ return atomic_fetch_${FNKEY}_explicit( this, __m__, __x__ ); }

EOF
done
done

echo impatomic.h methods pointer fetch
cat <<EOF >>impatomic.h

template< typename T >
T* atomic<T*>::load( memory_order __x__ ) volatile
{ return static_cast<T*>( atomic_address::load( __x__ ) ); }

template< typename T >
T* atomic<T*>::swap( T* __v__, memory_order __x__ ) volatile
{ return static_cast<T*>( atomic_address::swap( __v__, __x__ ) ); }

template< typename T >
bool atomic<T*>::compare_swap
( T*& __r__, T* __v__, memory_order __x__, memory_order __y__) volatile
{ return atomic_address::compare_swap( *reinterpret_cast<void**>( &__r__ ),
               static_cast<void*>( __v__ ), __x__, __y__ ); }
//{ return _ATOMIC_CMPSWP_( this, &__r__, __v__, __x__ ); }

template< typename T >
bool atomic<T*>::compare_swap
( T*& __r__, T* __v__, memory_order __x__ ) volatile
{ return compare_swap( __r__, __v__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

template< typename T >
T* atomic<T*>::fetch_add( ptrdiff_t __v__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, sizeof(T) * __v__, __x__ ); }

template< typename T >
T* atomic<T*>::fetch_sub( ptrdiff_t __v__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, sizeof(T) * __v__, __x__ ); }

EOF

cat <<EOF >>impatomic.h

#endif

EOF
echo impatomic.h close namespace
cat <<EOF >>impatomic.h

#ifdef __cplusplus
} // namespace std
#endif

EOF
echo stdatomic.h
cat <<EOF >stdatomic.h

#include "impatomic.h"

#ifdef __cplusplus

EOF

for TYPEKEY in flag bool address ${INTEGERS} ${CHARACTERS}
do
cat <<EOF >>stdatomic.h

using std::atomic_${TYPEKEY};

EOF
done

cat <<EOF >>stdatomic.h

using std::atomic;
using std::memory_order;
using std::memory_order_relaxed;
using std::memory_order_acquire;
using std::memory_order_release;
using std::memory_order_acq_rel;
using std::memory_order_seq_cst;

#endif

EOF

echo cstdatomic
cat <<EOF >cstdatomic

#include "impatomic.h"

EOF
echo n2427.c include
cat <<EOF >n2427.c

#include "stdatomic.h"

EOF
echo n2427.c flag
cat <<EOF >>n2427.c

atomic_flag af = ATOMIC_FLAG_INIT;

void flag_example( void )
{
    if ( ! atomic_flag_test_and_set_explicit( &af, memory_order_acquire ) )
        atomic_flag_clear_explicit( &af, memory_order_release );
#ifdef __cplusplus
    if ( ! af.test_and_set() )
        af.clear();
#endif
}

EOF
echo n2427.c lazy
cat <<EOF >>n2427.c

atomic_bool lazy_ready = { false };
atomic_bool lazy_assigned = { false };
int lazy_value;

#ifdef __cplusplus

int lazy_example_strong_cpp( void )
{
    if ( ! lazy_ready.load() ) {
        /* the value is not yet ready */
        if ( lazy_assigned.swap( true ) ) {
            /* initialization assigned to another thread; wait */
            while ( ! lazy_ready.load() );
        }
        else {
            lazy_value = 42;
            lazy_ready = true;
        }
    }
    return lazy_value;
}

#endif

int lazy_example_weak_c( void )
{
    if ( ! atomic_load_explicit( &lazy_ready, memory_order_acquire ) ) {
        if ( atomic_swap_explicit( &lazy_assigned, true,
                                   memory_order_relaxed ) ) {
            while ( ! atomic_load_explicit( &lazy_ready,
                                            memory_order_acquire ) );
        }
        else {
            lazy_value = 42;
            atomic_store_explicit( &lazy_ready, true, memory_order_release );
        }
    }
    return lazy_value;
}

#ifdef __cplusplus

int lazy_example_fence_cpp( void )
{
    if ( lazy_ready.load( memory_order_relaxed ) )
        lazy_ready.fence( memory_order_acquire );
    else if ( lazy_assigned.swap( true, memory_order_relaxed ) ) {
        while ( ! lazy_ready.load( memory_order_relaxed ) );
        lazy_ready.fence( memory_order_acquire );
    }
    else {
        lazy_value = 42;
        lazy_ready.store( true, memory_order_release );
    }
    return lazy_value;
}

#endif

EOF
echo n2427.c integer
cat <<EOF >>n2427.c

atomic_ulong volatile aulv = { 0 };
atomic_ulong auln = { 1 };
#ifdef __cplusplus
atomic< unsigned long > taul CPP0X( { 3 } );
#endif

void integer_example( void )
{
    atomic_ulong a = { 3 };
    unsigned long x = atomic_load( &auln );
    atomic_store_explicit( &aulv, x, memory_order_release );
    unsigned long y = atomic_fetch_add_explicit( &aulv, 1,
                                                 memory_order_relaxed );
    unsigned long z = atomic_fetch_xor( &auln, 4 );
#ifdef __cplusplus
    // x = auln; // implicit conversion disallowed
    x = auln.load();
    aulv = x;
    auln += 1;
    aulv ^= 4;
    // auln = aulv; // uses a deleted operator
    aulv -= auln++;
    auln |= --aulv;
    aulv &= 7;
    atomic_store_explicit( &taul, 7, memory_order_release );
    x = taul.load( memory_order_acquire );
    y = atomic_fetch_add_explicit( & taul, 1, memory_order_acquire );
    z = atomic_fetch_xor( & taul, 4 );
    x = taul.load();
    // auln = taul; // uses a deleted operator
    // taul = aulv; // uses a deleted operator
    taul = x;
    taul += 1;
    taul ^= 4;
    taul -= taul++;
    taul |= --taul;
    taul &= 7;
#endif
}

EOF
echo n2427.c event
cat <<EOF >>n2427.c

#ifdef __cplusplus

struct event_counter
{
    void inc() { au.fetch_add( 1, memory_order_relaxed ); }
    unsigned long get() { au.load( memory_order_relaxed ); }
    atomic_ulong au;
};
event_counter ec = { 0 };

void generate_events()
{
    ec.inc();
    ec.inc();
    ec.inc();
}

int read_events()
{
    return ec.get();
}

int event_example()
{
    generate_events(); // possibly in multiple threads
    // join all other threads, ensuring that final value is written
    return read_events();
}

#endif

EOF
echo n2427.c list
cat <<EOF >>n2427.c

#ifdef __cplusplus

struct data;
struct node
{
    node* next;
    data* value;
};

atomic< node* > head CPP0X( { (node*)0 } );

void list_example_strong( data* item )
{
    node* candidate = new node;
    candidate->value = item;
    candidate->next = head.load();
    while ( ! head.compare_swap( candidate->next, candidate ) );
}

void list_example_weak( data* item )
{
    node* candidate = new node;
    candidate->value = item;
    candidate->next = head.load( memory_order_relaxed );
    while ( ! head.compare_swap( candidate->next, candidate,
                                 memory_order_release, memory_order_relaxed ) );
}

#endif

EOF
echo n2427.c update
cat <<EOF >>n2427.c

#if ATOMIC_INTEGRAL_LOCK_FREE <= 1
atomic_flag pseudo_mutex = ATOMIC_FLAG_INIT;
unsigned long regular_variable = 1;
#endif
#if ATOMIC_INTEGRAL_LOCK_FREE >= 1
atomic_ulong atomic_variable = { 1 };
#endif

void update()
{
#if ATOMIC_INTEGRAL_LOCK_FREE == 1
    if ( atomic_is_lock_free( &atomic_variable ) ) {
#endif
#if ATOMIC_INTEGRAL_LOCK_FREE > 0
        unsigned long full = atomic_load( atomic_variable );
        unsigned long half = full / 2;
        while ( ! atomic_compare_swap( &atomic_variable, &full, half ) )
            half = full / 2;
#endif
#if ATOMIC_INTEGRAL_LOCK_FREE == 1
    } else {
#endif
#if ATOMIC_INTEGRAL_LOCK_FREE < 2
        __atomic_flag_wait__( &pseudo_mutex );
        regular_variable /= 2 ;
        atomic_flag_clear( &pseudo_mutex );
#endif
#if ATOMIC_INTEGRAL_LOCK_FREE == 1
    }
#endif
}

EOF
echo n2427.c main
cat <<EOF >>n2427.c

int main()
{
}

EOF
